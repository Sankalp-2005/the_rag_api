"""
RAG Ingestion and Embedding Service
===================================
A high-performance FastAPI service designed to ingest documents, convert them 
to Markdown, segment the text into overlapping chunks, generate dense vector 
embeddings using an OpenAI-compatible API, and store them in Qdrant.

Request Processing Pipeline
---------------------------
1. **Validation**: Enforce extension whitelist and file size constraints upfront.
2. **Streaming I/O**: Stream files concurrently to temporary storage to ensure
   O(1) memory complexity relative to file size.
3. **Markdown Conversion**: Parse raw files into Markdown formatted text in parallel.
4. **Document Generation**: Construct memory-mapped Document models for chunking.
5. **Semantic Chunking**: Split documents into overlapping segments using a 
   Recursive Character Text Splitter.
6. **Idempotence & Deduplication**: Generate a deterministic UUID-v5 for each chunk 
   based on its content hash. Query Qdrant to filter out pre-existing IDs.
7. **Vector Ingestion**: Generate embeddings for new chunks and upsert them to Qdrant.

Deduplication Strategy
----------------------
To minimize vector database storage footprint and API costs, every chunk is assigned 
a deterministic UUID-v5 derived from the SHA-256 hash of its text. Prior to 
embedding (the most computationally expensive step), the database is queried using 
these UUIDs. Chunks that are already indexed are skipped, making subsequent 
uploads of duplicate or heavily overlapping documents highly efficient.

Performance Design Notes
------------------------
- **Singleton Lifecycle**: Database connections, embedding model instances, and 
  text splitters are initialized once during the FastAPI startup lifecycle.
- **Auto-Provisioning**: The target Qdrant collection is automatically verified 
  and provisioned on initial boot.
- **Concurrent I/O**: Asynchronous file ingestion scales concurrently using 
  asyncio event loops.
- **Flat Memory Footprint**: Ingestion utilizes streaming chunks (1 MB slices) 
  to maintain stable memory usage regardless of file scale.
- **Resource Safety**: Robust exception handling ensures temporary filesystem 
  resources are cleaned up under all failure modes.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Set

from any_to_markdown import get_markdown
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ===========================================================================
# Configuration & Logging
# ===========================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum allowed file size limit (200 MB) matching the downstream parser constraints.
MAX_FILE_SIZE_BYTES: int = 200 * 1024 * 1024

# Target Qdrant collection name, configurable via environment variables.
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "RAG")

# Set of supported file extensions allowed for ingestion.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt", ".md", ".json", ".csv", ".pdf", ".docx",
        ".xls", ".xlsx", ".pptx", ".ipynb",
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
        ".mp3", ".wav", ".m4a", ".mp4",
        ".py", ".js", ".ts", ".go", ".rs", ".java", ".rb",
        ".cpp", ".c", ".h", ".hpp", ".php", ".sh", ".sql",
        ".yaml", ".yml", ".xml", ".html", ".htm", ".css",
    }
)

# Buffer size (1 MB) used for streaming file uploads to disk.
_UPLOAD_CHUNK_BYTES: int = 1024 * 1024  # 1 MB


# ===========================================================================
# Application State Singletons
# ===========================================================================

class _AppState:
    """
    State container for heavy, reusable resources initialized at application startup.

    Attributes
    ----------
    embeddings : OpenAIEmbeddings
        The OpenAI-compatible embedding model client.
    splitter : RecursiveCharacterTextSplitter
        The pre-configured recursive character text splitter.
    qdrant_client : QdrantClient
        The persistent TCP connection client to the Qdrant cluster.
    vector_store : QdrantVectorStore
        A thin LangChain abstraction layer wrapping the Qdrant client.
    """

    embeddings: OpenAIEmbeddings
    splitter: RecursiveCharacterTextSplitter
    qdrant_client: QdrantClient
    vector_store: QdrantVectorStore


_state = _AppState()


# ===========================================================================
# Lifespan Hook Management (Startup / Shutdown)
# ===========================================================================

def _ensure_collection_exists(
    client: QdrantClient,
    name: str,
    embeddings: OpenAIEmbeddings,
) -> None:
    """
    Verify the existence of the target Qdrant collection, creating it if necessary.

    Performs an existence check on the collection. If absent, a short probe string 
    is embedded to dynamically determine the vector dimensionality of the configured 
    model. The collection is then created utilizing Cosine similarity.

    Parameters
    ----------
    client : QdrantClient
        An active, connected Qdrant client instance.
    name : str
        The name of the target collection to verify or create.
    embeddings : OpenAIEmbeddings
        The embedding client utilized to resolve vector dimension requirements.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        logger.info("Qdrant collection '%s' already exists.", name)
        return

    # Embed a short probe string to discover vector dimension at runtime
    # so this code stays model-agnostic.
    probe_vector = embeddings.embed_query("dimension probe")
    vector_dim = len(probe_vector)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    logger.info(
        "Created Qdrant collection '%s' (dim=%d, distance=cosine).",
        name,
        vector_dim,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for application startup and shutdown hook registry.

    Initializes expensive singletons (embeddings, text splitter, Qdrant connections) 
    at boot, ensuring resources are shared efficiently across incoming requests 
    and closed gracefully upon termination.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance registering the lifespan hooks.
    """
    logger.info("Initialising RAG Embedding Service…")

    # Validate required environment configurations
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    embedding_base_url = os.getenv("BASE_URL_FOR_EMBEDDING_MODEL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    missing = []
    if not embedding_model:
        missing.append("EMBEDDING_MODEL_NAME")
    if not embedding_base_url:
        missing.append("BASE_URL_FOR_EMBEDDING_MODEL")
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    # Initialize embedding client (maintains underlying HTTP session pool)
    _state.embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url,
    )

    # Initialize the text splitter (stateless, safe for concurrent requests)
    _state.splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # Ordered from coarsest to finest so the splitter preserves semantic
        # boundaries as long as the chunk-size limit allows.
        separators=[
            "\n\n\n",  # large section / chapter breaks
            "\n\n",    # paragraphs
            "\n",      # lines
            " ",       # words
            "",        # character-level fallback
        ],
    )

    # Establish connection to the Qdrant cluster
    _state.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify or provision the target vector collection
    _ensure_collection_exists(
        _state.qdrant_client, COLLECTION_NAME, _state.embeddings
    )

    # Instantiate the vector store abstraction
    _state.vector_store = QdrantVectorStore(
        client=_state.qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=_state.embeddings,
    )

    logger.info("Startup complete — ready to serve requests.")
    yield

    # Gracefully close connections on shutdown
    _state.qdrant_client.close()
    logger.info("Qdrant connection closed. Shutting down.")


# ===========================================================================
# FastAPI App Definition
# ===========================================================================

app = FastAPI(
    title="THE RAG API",
    description=(
        "Upload files of any supported type. The service converts them to "
        "Markdown, chunks the text, deduplicates against existing vectors, "
        "embeds only new chunks, and stores them in Qdrant."
    ),
    lifespan=lifespan
)


# ===========================================================================
# Pydantic Schemas
# ===========================================================================

class FileDetail(BaseModel):
    """Represents the processing outcome and status of an individual file."""

    input: str
    status: str
    error: str | None = None


class EmbeddingResponse(BaseModel):
    """
    Response schema summarizing the execution of the ingestion and embedding pipeline.

    Attributes
    ----------
    status : str
        Execution status, typically "completed".
    total_files : int
        The total number of files received in the payload.
    successful : int
        The number of files successfully parsed and processed.
    failed : int
        The number of files that failed during the parsing or ingestion stages.
    total_chunks : int
        The total number of text segments generated by the splitter.
    new_chunks : int
        The number of new text segments successfully vectorized and stored.
    skipped_chunks : int
        The number of duplicate text segments skipped during deduplication.
    details : list[FileDetail]
        A detailed list containing the execution status and errors for each file.
    """

    status: str
    total_files: int
    successful: int
    failed: int
    total_chunks: int
    new_chunks: int
    skipped_chunks: int
    details: List[FileDetail]


# ===========================================================================
# Helper Functions - Validation & Storage
# ===========================================================================

def _validate_upload(upload: UploadFile) -> None:
    """
    Perform pre-ingestion validation on an uploaded file's metadata.

    Parameters
    ----------
    upload : UploadFile
        The uploaded file to validate.

    Raises
    ------
    HTTPException
        If the filename is empty (400) or contains an unsupported extension (415).
    """
    filename = upload.filename or ""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more uploaded files is missing a filename.",
        )

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"'{filename}' has unsupported extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    """
    Stream an uploaded file to a local destination using a flat memory buffer.

    Enforces size limits incrementally during the streaming process to avoid 
    unnecessary disk I/O and memory consumption for oversized payloads.

    Parameters
    ----------
    upload : UploadFile
        The FastAPI upload resource stream.
    dest : Path
        The local destination path to write the contents.

    Raises
    ------
    HTTPException
        If the file exceeds size constraints (413) or encounters disk write errors (500).
    """
    written = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"'{upload.filename}' exceeds the "
                            f"{MAX_FILE_SIZE_BYTES // (1024 ** 2)} MB limit."
                        ),
                    )
                fh.write(chunk)
    except HTTPException:
        raise  # re-raise our own size error unchanged
    except OSError as exc:
        logger.exception("I/O error while saving '%s'", upload.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save '{upload.filename}': {exc}",
        ) from exc


# ===========================================================================
# Helper Functions - Deduplication
# ===========================================================================

def _content_id(text: str) -> str:
    """
    Generate a deterministic UUID-v5 for a text segment to enable id-based deduplication.

    Computes a SHA-256 digest of the UTF-8 encoded text and hashes it using 
    a UUID-v5 namespace to generate a globally unique, content-addressable identifier.

    Parameters
    ----------
    text : str
        The raw text content of the document segment.

    Returns
    -------
    str
        A deterministic UUID-v5 string representation.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, digest))


def _find_existing_ids(
    client: QdrantClient,
    collection: str,
    candidate_ids: List[str],
) -> Set[str]:
    """
    Query the vector database to check which candidate IDs are already indexed.

    Performs a lightweight existence query retrieving only the point IDs to 
    minimize database transmission payloads.

    Parameters
    ----------
    client : QdrantClient
        An active, connected Qdrant client.
    collection : str
        The name of the target collection.
    candidate_ids : list[str]
        A list of UUID strings to query in the index.

    Returns
    -------
    set[str]
        A set of existing UUID strings retrieved from the database.
    """
    if not candidate_ids:
        return set()

    try:
        existing_points = client.retrieve(
            collection_name=collection,
            ids=candidate_ids,
            with_payload=False,
            with_vectors=False,
        )
        found = {str(point.id) for point in existing_points}
        logger.info(
            "Dedup check: %d / %d chunk(s) already exist in '%s'.",
            len(found),
            len(candidate_ids),
            collection,
        )
        return found

    except Exception as exc:
        # If the lookup itself fails (network blip, etc.) we log a warning
        # and proceed without dedup — better to insert a few duplicates than
        # to reject the entire request.
        logger.warning(
            "Dedup lookup failed (%s); proceeding without deduplication.",
            exc,
        )
        return set()


# ===========================================================================
# Endpoint Definition
# ===========================================================================

@app.post(
    "/store-embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest files into the Qdrant vector store",
    response_description="Pipeline execution summary detailing conversion and deduplication metrics.",
)
async def store_embeddings(files: List[UploadFile] = File(...)):
    """
    Process, segment, deduplicate, and index uploaded files into the Qdrant vector database.

    Runs files through a comprehensive pipeline: validating formats, streaming to disk,
    converting to Markdown text, splitting into semantically structured chunks, checking
    against pre-existing points using UUID content hashing, generating embeddings for new
    segments, and upserting vectors to Qdrant.

    Parameters
    ----------
    files : list[UploadFile]
        A list of files transmitted via multipart/form-data.

    Returns
    -------
    EmbeddingResponse
        An object containing processing statistics and per-file validation/conversion details.

    Raises
    ------
    HTTPException
        - 400: Bad Request (no files provided or missing filenames).
        - 413: Payload Too Large (a file exceeds the 200 MB limit).
        - 415: Unsupported Media Type (invalid extension).
        - 422: Unprocessable Entity (all files failed conversion).
        - 500: Internal Server Error (unexpected pipeline failure).
        - 502: Bad Gateway (upstream database write failure).
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided.",
        )

    # Validate upload metadata prior to performing disk write operations
    for upload in files:
        _validate_upload(upload)

    # Stream files concurrently to temporary storage to isolate session files
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))

    try:
        # Generate destination mapping; append indices to prevent namespace collisions
        dest_map = {
            upload: tmp_dir / (upload.filename or f"file_{i}")
            for i, upload in enumerate(files)
        }

        # Save upload streams concurrently to optimize network and disk throughput
        await asyncio.gather(
            *(_save_upload(upload, dest) for upload, dest in dest_map.items())
        )
        saved_paths = list(dest_map.values())

        logger.info(
            "Saved %d file(s) to temp dir '%s'. Starting conversion…",
            len(saved_paths),
            tmp_dir,
        )

        # Parse files to Markdown; conversion is parallelized internally
        results = await get_markdown(saved_paths)

    except HTTPException:
        raise  # surface our own validation / size errors unchanged
    except Exception as exc:
        logger.exception("Unexpected error during file save or markdown conversion.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during file processing: {exc}",
        ) from exc
    finally:
        # Ensure temporary disk space is cleaned up under all execution paths
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Cleaned up temp dir '%s'.", tmp_dir)

    # Construct memory-mapped Document models using parsed content
    docs: List[Document] = []
    details: List[FileDetail] = []

    for r in results:
        if r.ok and r.content:
            docs.append(
                Document(
                    page_content=r.content,
                    metadata={"source": r.input},
                )
            )
            details.append(FileDetail(input=r.input, status="success"))
        else:
            logger.warning("Conversion failed for '%s': %s", r.input, r.error)
            details.append(
                FileDetail(input=r.input, status=r.status, error=r.error)
            )

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "All files failed conversion — nothing was stored.",
                "details": [d.model_dump() for d in details],
            },
        )

    # Segment documents into overlapping chunks using the pre-configured splitter
    try:
        chunks = _state.splitter.split_documents(docs)
        logger.info("Split %d document(s) into %d chunk(s).", len(docs), len(chunks))
    except Exception as exc:
        logger.exception("Text splitting failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text splitting failed: {exc}",
        ) from exc

    # Deduplicate segments to prevent redundant embedding generation and storage
    # Calculate deterministic UUIDs from chunk text content
    chunk_ids = [_content_id(chunk.page_content) for chunk in chunks]

    # Query database for existing points
    existing_ids = _find_existing_ids(
        _state.qdrant_client, COLLECTION_NAME, chunk_ids
    )

    # Filter out pre-existing segments
    new_chunks: List[Document] = []
    new_ids: List[str] = []
    for chunk, cid in zip(chunks, chunk_ids):
        if cid not in existing_ids:
            # Store content hash as metadata for downstream auditability
            chunk.metadata["content_hash"] = cid
            new_chunks.append(chunk)
            new_ids.append(cid)

    skipped_count = len(chunks) - len(new_chunks)

    if skipped_count:
        logger.info(
            "Dedup: %d chunk(s) are new, %d already exist — skipping duplicates.",
            len(new_chunks),
            skipped_count,
        )

    # Vectorize and ingest new segments to the database
    if new_chunks:
        try:
            _state.vector_store.add_documents(new_chunks, ids=new_ids)
            logger.info(
                "Successfully embedded and stored %d new chunk(s) in Qdrant.",
                len(new_chunks),
            )
        except Exception as exc:
            logger.exception("Failed to upsert embeddings into Qdrant.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Qdrant upsert failed: {exc}",
            ) from exc
    else:
        logger.info("All %d chunk(s) already exist — nothing to upload.", len(chunks))

    # Construct execution metrics response
    successful = sum(1 for d in details if d.status == "success")
    failed = len(details) - successful

    return EmbeddingResponse(
        status="completed",
        total_files=len(results),
        successful=successful,
        failed=failed,
        total_chunks=len(chunks),
        new_chunks=len(new_chunks),
        skipped_chunks=skipped_count,
        details=details,
    )

import os
import glob
from pathlib import Path
from ingestor import create_ingestor
from app.utils.logger import setup_logging
from qdrant_client import QdrantClient

logger = setup_logging("Chunk and ingest")

qdrant_url = os.environ.get("QDRANT_URL")
data_dir = "./data"
collection_name = "math_philosophy"
batch_size = 32
max_workers = 4
file_patterns = ["*.pdf", "*.txt"]

client = QdrantClient(qdrant_url)

try:
    client.delete_collection(collection_name)
    logger.info(f"Deleted collection: {collection_name}")
except Exception as e:
    logger.error(f"Failed to delete: {e}")

docs = []
for pattern in file_patterns:
    docs.extend(glob.glob(os.path.join(data_dir, pattern)))

logger.info(f"Found {len(docs)} documents to ingest")

ingestor = create_ingestor(
    qdrant_url=qdrant_url,
    collection_name=collection_name,
    batch_size=batch_size,
    max_workers=max_workers
)

vector_store = ingestor.ingest_documents(docs)

logger.info(f"Vector store stats: {ingestor.get_vector_store_stats()}")


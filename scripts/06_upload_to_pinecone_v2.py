#!/usr/bin/env python3
"""
06_upload_to_pinecone_v2.py
Upload YouTube transcript embeddings to Pinecone vector database.

================================================================================
OVERVIEW
================================================================================

This script is Step 6 (final step) of the YouTube AI Semantic Search pipeline.
It takes the embedding vectors generated in Step 5 and uploads them to Pinecone,
making them searchable via semantic/similarity search.

What is Pinecone?
- A managed vector database optimized for similarity search
- Stores embedding vectors with associated metadata
- Enables fast nearest-neighbor search across millions of vectors
- Supports namespaces to organize different content types

================================================================================
HOW IT WORKS
================================================================================

1. LOAD EMBEDDINGS
   - Reads youtube_embeddings.json from Step 05
   - Each chunk has a 1536-dimension embedding vector + metadata

2. PREPARE VECTORS
   - Formats each chunk for Pinecone's expected structure
   - Truncates metadata to stay within Pinecone's 40KB limit
   - Adds content_type field for filtering

3. BATCH UPLOAD
   - Groups vectors into batches (default: 100 per batch)
   - Upserts to Pinecone (insert or update if ID exists)
   - Uses namespace to separate from other content

4. VERIFY
   - Reports final vector count in index
   - Optional test search to verify functionality

================================================================================
PINECONE STRUCTURE
================================================================================

Index: Configured in config.py (e.g., "mohler-ai")
Namespace: "youtube" (separates from other content types)

Each vector contains:
- id: Unique chunk identifier (yt-{video_id}-{chunk_index})
- values: 1536-dimension embedding array
- metadata: Searchable/filterable fields

Metadata fields:
- video_id, chunk_index: Core identifiers
- text: Transcript text (truncated to 1000 chars)
- start_time, end_time: Seconds for deep linking
- start_timestamp, end_timestamp: Human-readable times
- video_title, channel, thumbnail_url: Display info
- youtube_url: Direct link with timestamp
- content_type: "youtube_transcript" (for filtering)

================================================================================
INPUT FORMAT (from Script 05)
================================================================================

data/embeddings/youtube_embeddings.json:
{
    "chunks": [
        {
            "chunk_id": "yt-abc123-0000",
            "embedding": [0.0123, -0.0456, ...],  // 1536 floats
            "video_id": "abc123",
            "text": "...",
            "video_title": "...",
            ...
        }
    ]
}

================================================================================
USAGE
================================================================================

Basic upload (all embeddings):
    python 06_upload_to_pinecone_v2.py

Limit upload (for testing):
    python 06_upload_to_pinecone_v2.py --limit 100

Dry run (preview without uploading):
    python 06_upload_to_pinecone_v2.py --dry-run

Delete all vectors in namespace (careful!):
    python 06_upload_to_pinecone_v2.py --delete-all

Test search after upload:
    python 06_upload_to_pinecone_v2.py --test
    python 06_upload_to_pinecone_v2.py --test --query "your search query"

================================================================================
REQUIREMENTS
================================================================================

Prerequisites:
    - Python 3.8+
    - config.py in project root (created by yt_ai_search_setup.sh)
    - Completed Step 05 (youtube_embeddings.json exists)
    - Pinecone API key in .env file
    - Pinecone index already created

Dependencies:
    - pinecone-client>=3.0.0

Pinecone Setup:
    1. Create account at https://app.pinecone.io
    2. Create index with:
       - Dimensions: 1536
       - Metric: cosine
       - Cloud: AWS or GCP (starter tier is free)
    3. Get API key from console
    4. Add to .env: PINECONE_API_KEY=...

================================================================================
NAMESPACES
================================================================================

Pinecone namespaces allow multiple content types in one index:
- "youtube": YouTube transcript chunks (this pipeline)
- "commentary": Bible commentary (other content)
- etc.

Benefits:
- Single index, lower cost
- Can search across all or filter by namespace
- Easy to delete/rebuild one namespace without affecting others

================================================================================
METADATA LIMITS
================================================================================

Pinecone has metadata size limits:
- Total metadata per vector: 40KB
- Recommended string length: <10KB each

This script truncates:
- text: 1000 characters (with "..." suffix if truncated)
- video_title: 200 characters

================================================================================
PINECONE SETTINGS (from config.py)
================================================================================

PINECONE_INDEX = "mohler-ai"      # Your index name
PINECONE_NAMESPACE = "youtube"    # Namespace for YouTube content
PINECONE_BATCH_SIZE = 100         # Vectors per upsert call

================================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIG IMPORTS
# =============================================================================

from config import (
    CHANNEL_HANDLE,
    CHANNEL_DISPLAY_NAME,
    PINECONE_API_KEY,
    OPENAI_API_KEY,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
    PINECONE_BATCH_SIZE,
    EMBEDDINGS_FILE,
    LOGS_DIR,
    EMBEDDING_MODEL,
    ensure_directories,
    get_log_file,
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging to file and console."""
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(get_log_file('06_upload_to_pinecone')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# PINECONE HELPERS
# =============================================================================

def get_pinecone_index():
    """Initialize Pinecone and return index."""
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found!\n"
            "Add to .env file: PINECONE_API_KEY=...\n"
            "Get key from: https://app.pinecone.io"
        )
    
    # Import here to avoid error if not installed
    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError(
            "Pinecone package not installed!\n"
            "Run: pip install pinecone-client"
        )
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    
    return index


def prepare_vector(chunk):
    """
    Prepare a chunk for Pinecone upload.
    
    Pinecone metadata has size limits:
    - Total metadata size: 40KB per vector
    - Individual string values: should be under 10KB
    """
    
    # Truncate text for metadata (keep under 1000 chars for safety)
    text_preview = chunk.get("text", "")[:1000]
    if len(chunk.get("text", "")) > 1000:
        text_preview += "..."
    
    return {
        "id": chunk["chunk_id"],
        "values": chunk["embedding"],
        "metadata": {
            # Core identifiers
            "video_id": chunk["video_id"],
            "chunk_index": chunk["chunk_index"],
            
            # Text content (truncated for metadata limits)
            "text": text_preview,
            
            # Timestamps for deep linking
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "start_timestamp": chunk["start_timestamp"],
            "end_timestamp": chunk["end_timestamp"],
            "duration_seconds": chunk["duration_seconds"],
            
            # Video metadata
            "video_title": chunk.get("video_title", "")[:200],
            "channel": chunk.get("channel", CHANNEL_DISPLAY_NAME),
            "video_duration_seconds": chunk.get("video_duration_seconds", 0),
            "thumbnail_url": chunk.get("thumbnail_url", ""),
            
            # Links
            "youtube_url": chunk["youtube_url"],
            "video_url": chunk["video_url"],
            
            # For filtering
            "content_type": "youtube_transcript"
        }
    }


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def upload_to_pinecone(limit=None, dry_run=False):
    """Upload embeddings to Pinecone."""
    
    logger.info("=" * 60)
    logger.info("Starting Pinecone Upload")
    logger.info(f"Channel: {CHANNEL_DISPLAY_NAME} ({CHANNEL_HANDLE})")
    logger.info("=" * 60)
    
    logger.info(f"Pinecone settings:")
    logger.info(f"  Index: {PINECONE_INDEX}")
    logger.info(f"  Namespace: {PINECONE_NAMESPACE}")
    logger.info(f"  Batch size: {PINECONE_BATCH_SIZE}")
    
    # Load embeddings
    if not EMBEDDINGS_FILE.exists():
        logger.error(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        logger.error("Run 05_generate_embeddings_v2.py first")
        return
    
    logger.info(f"Loading embeddings from {EMBEDDINGS_FILE}")
    with open(EMBEDDINGS_FILE, 'r') as f:
        embeddings_data = json.load(f)
    
    chunks = embeddings_data.get("chunks", [])
    logger.info(f"Loaded {len(chunks)} chunks with embeddings")
    
    # Apply limit
    if limit:
        chunks = chunks[:limit]
        logger.info(f"Limit mode: uploading first {limit} chunks")
    
    if not chunks:
        logger.info("No chunks to upload!")
        return
    
    if dry_run:
        logger.info("")
        logger.info("DRY RUN - No actual upload will occur")
        logger.info(f"Would upload {len(chunks)} vectors to:")
        logger.info(f"  Index: {PINECONE_INDEX}")
        logger.info(f"  Namespace: {PINECONE_NAMESPACE}")
        
        # Show sample
        sample = prepare_vector(chunks[0])
        logger.info("")
        logger.info("Sample vector:")
        logger.info(f"  ID: {sample['id']}")
        logger.info(f"  Dimensions: {len(sample['values'])}")
        logger.info(f"  Metadata keys: {list(sample['metadata'].keys())}")
        return
    
    # Initialize Pinecone
    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX}")
    try:
        index = get_pinecone_index()
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return
    
    # Get current stats
    stats = index.describe_index_stats()
    logger.info(f"Current index stats: {stats.get('total_vector_count', 0)} total vectors")
    
    if PINECONE_NAMESPACE in stats.get("namespaces", {}):
        ns_stats = stats["namespaces"][PINECONE_NAMESPACE]
        logger.info(f"Current '{PINECONE_NAMESPACE}' namespace: {ns_stats.get('vector_count', 0)} vectors")
    
    # Upload in batches
    total_batches = (len(chunks) + PINECONE_BATCH_SIZE - 1) // PINECONE_BATCH_SIZE
    stats_upload = {"uploaded": 0, "failed": 0}
    
    start_time = datetime.now()
    
    for batch_idx in range(0, len(chunks), PINECONE_BATCH_SIZE):
        batch = chunks[batch_idx:batch_idx + PINECONE_BATCH_SIZE]
        batch_num = batch_idx // PINECONE_BATCH_SIZE + 1
        
        try:
            # Prepare vectors
            vectors = [prepare_vector(chunk) for chunk in batch]
            
            # Upsert to Pinecone
            index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
            
            stats_upload["uploaded"] += len(vectors)
            logger.info(f"[Batch {batch_num}/{total_batches}] ✓ Uploaded {len(vectors)} vectors")
            
        except Exception as e:
            logger.error(f"[Batch {batch_num}/{total_batches}] ✗ Error: {e}")
            stats_upload["failed"] += len(batch)
    
    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Get updated stats
    final_stats = index.describe_index_stats()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time elapsed: {elapsed:.2f} seconds")
    logger.info(f"Vectors uploaded: {stats_upload['uploaded']}")
    logger.info(f"Failed: {stats_upload['failed']}")
    logger.info(f"Index: {PINECONE_INDEX}")
    logger.info(f"Namespace: {PINECONE_NAMESPACE}")
    logger.info(f"Total vectors in index: {final_stats.get('total_vector_count', 0)}")
    
    if PINECONE_NAMESPACE in final_stats.get("namespaces", {}):
        ns_count = final_stats["namespaces"][PINECONE_NAMESPACE].get("vector_count", 0)
        logger.info(f"Vectors in '{PINECONE_NAMESPACE}' namespace: {ns_count}")


def delete_namespace():
    """Delete all vectors in the configured namespace."""
    logger.warning(f"Deleting all vectors in namespace '{PINECONE_NAMESPACE}'")
    
    try:
        index = get_pinecone_index()
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return
    
    # Get current count
    stats = index.describe_index_stats()
    if PINECONE_NAMESPACE in stats.get("namespaces", {}):
        count = stats["namespaces"][PINECONE_NAMESPACE].get("vector_count", 0)
        logger.info(f"Current vectors in namespace: {count}")
    
    # Delete all in namespace
    index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    
    logger.info(f"✓ Deleted all vectors in namespace '{PINECONE_NAMESPACE}'")


def test_search(query=None):
    """Test search in the configured namespace."""
    
    # Default query uses channel name
    if query is None:
        query = f"What does {CHANNEL_DISPLAY_NAME} say about faith?"
    
    logger.info(f"Testing search with query: {query}")
    
    # Validate OpenAI key for generating query embedding
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found - needed to generate query embedding")
        return
    
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI package not installed!")
        logger.error("Run: pip install openai")
        return
    
    # Generate query embedding
    logger.info(f"Generating query embedding with {EMBEDDING_MODEL}...")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Search Pinecone
    try:
        index = get_pinecone_index()
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        return
    
    logger.info(f"Searching in index '{PINECONE_INDEX}', namespace '{PINECONE_NAMESPACE}'...")
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE
    )
    
    matches = results.get("matches", [])
    
    if not matches:
        logger.info("No results found!")
        return
    
    logger.info(f"\nTop {len(matches)} results:")
    logger.info("-" * 60)
    
    for i, match in enumerate(matches, 1):
        meta = match.get("metadata", {})
        logger.info(f"\n{i}. Score: {match['score']:.4f}")
        logger.info(f"   Video: {meta.get('video_title', 'N/A')}")
        logger.info(f"   Time: {meta.get('start_timestamp', 'N/A')} - {meta.get('end_timestamp', 'N/A')}")
        logger.info(f"   Text: {meta.get('text', 'N/A')[:150]}...")
        logger.info(f"   Link: {meta.get('youtube_url', 'N/A')}")


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Upload YouTube transcript embeddings to Pinecone.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 06_upload_to_pinecone_v2.py                  # Upload all embeddings
    python 06_upload_to_pinecone_v2.py --limit 100      # Upload first 100 chunks
    python 06_upload_to_pinecone_v2.py --dry-run        # Preview without uploading
    python 06_upload_to_pinecone_v2.py --delete-all     # Delete all vectors in namespace
    python 06_upload_to_pinecone_v2.py --test           # Test search after upload
    python 06_upload_to_pinecone_v2.py --test --query "your query"
        """
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of vectors to upload (for testing)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )
    
    parser.add_argument(
        '--delete-all',
        action='store_true',
        help=f'Delete all vectors in the "{PINECONE_NAMESPACE}" namespace'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a test search after upload'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Custom query for test search (use with --test)'
    )
    
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    if args.delete_all:
        # Confirmation prompt for destructive action
        confirm = input(
            f"Are you sure you want to delete all vectors in '{PINECONE_NAMESPACE}' namespace?\n"
            f"Index: {PINECONE_INDEX}\n"
            f"Type 'yes' to confirm: "
        )
        if confirm.lower() == "yes":
            delete_namespace()
        else:
            logger.info("Cancelled")
    
    elif args.test:
        test_search(query=args.query)
    
    else:
        upload_to_pinecone(limit=args.limit, dry_run=args.dry_run)

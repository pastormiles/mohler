#!/usr/bin/env python3
"""
05_generate_embeddings_v2.py
Generate OpenAI embeddings for transcript chunks.

================================================================================
OVERVIEW
================================================================================

This script is Step 5 of the YouTube AI Semantic Search pipeline. It takes
chunked transcripts and generates vector embeddings using OpenAI's embedding
API, preparing them for upload to Pinecone.

What are embeddings?
- Embeddings convert text into numerical vectors (arrays of floats)
- Similar meanings produce similar vectors (close in vector space)
- This enables semantic search: find content by meaning, not just keywords
- Example: "automobile" and "car" have similar embeddings

================================================================================
HOW IT WORKS
================================================================================

1. LOAD CHUNKS
   - Reads all_chunks.json from Step 04
   - Each chunk has text content and metadata

2. BATCH PROCESSING
   - Groups chunks into batches (default: 100 per batch)
   - Sends batches to OpenAI API
   - Batching is more efficient than one-at-a-time

3. GENERATE EMBEDDINGS
   - Uses text-embedding-3-small model (1536 dimensions)
   - Embeds the "embedding_text" field (title + timestamp + content)
   - Returns vector of 1536 floats per chunk

4. SAVE RESULTS
   - youtube_embeddings.json: Chunks with embedding vectors added
   - embedding_progress.json: Track which chunks have been processed

================================================================================
COST ESTIMATION
================================================================================

Model: text-embedding-3-small
Pricing: $0.02 per 1M tokens (as of 2024)

Typical costs:
- 1,000 chunks (~75 words each): ~$0.003
- 10,000 chunks: ~$0.03
- 100,000 chunks: ~$0.30

The script estimates cost before processing and reports actual usage after.

================================================================================
INPUT FORMAT (from Script 04)
================================================================================

data/chunks/all_chunks.json:
{
    "chunks": [
        {
            "chunk_id": "yt-abc123-0000",
            "video_id": "abc123",
            "text": "The actual transcript text...",
            "embedding_text": "Video Title | 0:00\n\nThe actual transcript text...",
            "video_title": "Video Title",
            ...
        }
    ]
}

================================================================================
OUTPUT FORMAT
================================================================================

data/embeddings/youtube_embeddings.json:
{
    "created_at": "2025-01-15T10:30:00",
    "model": "text-embedding-3-small",
    "dimensions": 1536,
    "total_chunks": 5432,
    "total_videos": 287,
    "channel_handle": "@AlbertMohler",
    "channel_display_name": "Albert Mohler",
    "chunks": [
        {
            "chunk_id": "yt-abc123-0000",
            "video_id": "abc123",
            "text": "...",
            "embedding": [0.0123, -0.0456, 0.0789, ...],  // 1536 floats
            ...
        }
    ]
}

================================================================================
USAGE
================================================================================

Basic usage (process all chunks):
    python 05_generate_embeddings_v2.py

Limit processing (for testing):
    python 05_generate_embeddings_v2.py --limit 100

Incremental mode (only process new chunks):
    python 05_generate_embeddings_v2.py --incremental

Custom batch size:
    python 05_generate_embeddings_v2.py --batch-size 50

Combined:
    python 05_generate_embeddings_v2.py --incremental --limit 1000 --batch-size 50

================================================================================
REQUIREMENTS
================================================================================

Prerequisites:
    - Python 3.8+
    - config.py in project root (created by yt_ai_search_setup.sh)
    - Completed Step 04 (all_chunks.json exists)
    - OpenAI API key in .env file

Dependencies:
    - openai>=1.0.0

API Key Setup:
    1. Get API key from https://platform.openai.com/api-keys
    2. Add to .env file: OPENAI_API_KEY=sk-...

================================================================================
RATE LIMITING
================================================================================

OpenAI has rate limits based on your account tier:
- Free tier: 3 RPM (requests per minute), 200 RPD (requests per day)
- Tier 1+: Much higher limits

The script includes:
- Configurable delay between batches (RATE_LIMIT_DELAY)
- Automatic retry on rate limit errors (via OpenAI client)
- Progress saving every 10 batches (resume if interrupted)

================================================================================
EMBEDDING SETTINGS (from config.py)
================================================================================

EMBEDDING_MODEL = "text-embedding-3-small"   # Model name
EMBEDDING_DIMENSIONS = 1536                   # Vector dimensions
EMBEDDING_BATCH_SIZE = 100                    # Chunks per API call

================================================================================
"""

import argparse
import json
import logging
import sys
import time
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
    OPENAI_API_KEY,
    CHUNKS_FILE,
    EMBEDDINGS_DIR,
    EMBEDDINGS_FILE,
    EMBEDDINGS_PROGRESS_FILE,
    LOGS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
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
            logging.FileHandler(get_log_file('05_generate_embeddings')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# CONSTANTS
# =============================================================================

RATE_LIMIT_DELAY = 0.1  # Delay between batches (seconds)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_progress():
    """Load embedding progress."""
    if EMBEDDINGS_PROGRESS_FILE.exists():
        with open(EMBEDDINGS_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"embedded_chunk_ids": [], "failed_chunk_ids": []}


def save_progress(progress):
    """Save embedding progress."""
    with open(EMBEDDINGS_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def generate_embeddings_batch(texts, client):
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of strings to embed
        client: OpenAI client
    
    Returns:
        List of embedding vectors
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    
    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]
    return embeddings


def estimate_cost(chunks):
    """Estimate OpenAI API cost for embedding chunks."""
    total_chars = sum(len(c.get("embedding_text", c.get("text", ""))) for c in chunks)
    # Rough estimate: 4 chars per token
    estimated_tokens = total_chars / 4
    # text-embedding-3-small: $0.02 per 1M tokens
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02
    return estimated_tokens, estimated_cost


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_embeddings(limit=None, incremental=False, batch_size=None):
    """Generate embeddings for all chunks."""
    
    # Use default batch size from config if not specified
    if batch_size is None:
        batch_size = EMBEDDING_BATCH_SIZE
    
    logger.info("=" * 60)
    logger.info("Starting Embedding Generation")
    logger.info(f"Channel: {CHANNEL_DISPLAY_NAME} ({CHANNEL_HANDLE})")
    logger.info("=" * 60)
    
    logger.info(f"Embedding settings:")
    logger.info(f"  Model: {EMBEDDING_MODEL}")
    logger.info(f"  Dimensions: {EMBEDDING_DIMENSIONS}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Validate API key
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found!")
        logger.error("Add to .env file: OPENAI_API_KEY=sk-...")
        logger.error("Get key from: https://platform.openai.com/api-keys")
        return
    
    # Import OpenAI here to avoid import error if not installed
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("OpenAI package not installed!")
        logger.error("Run: pip install openai")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
    
    # Load chunks
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        logger.error("Run 04_chunk_transcripts_v2.py first")
        return
    
    with open(CHUNKS_FILE, 'r') as f:
        chunks_data = json.load(f)
    
    all_chunks = chunks_data.get("chunks", [])
    logger.info(f"Loaded {len(all_chunks)} chunks from {CHUNKS_FILE}")
    
    # Load progress
    progress = load_progress()
    
    # Filter chunks to process
    if incremental:
        already_embedded = set(progress.get("embedded_chunk_ids", []))
        chunks_to_process = [c for c in all_chunks if c["chunk_id"] not in already_embedded]
        logger.info(f"Incremental mode: {len(already_embedded)} already embedded")
    else:
        chunks_to_process = all_chunks
        progress = {"embedded_chunk_ids": [], "failed_chunk_ids": []}
    
    # Apply limit
    if limit:
        chunks_to_process = chunks_to_process[:limit]
        logger.info(f"Limit mode: processing first {limit} chunks")
    
    logger.info(f"Chunks to embed: {len(chunks_to_process)}")
    
    if not chunks_to_process:
        logger.info("No chunks to embed!")
        return
    
    # Estimate cost
    est_tokens, est_cost = estimate_cost(chunks_to_process)
    logger.info(f"Estimated tokens: {est_tokens:,.0f}")
    logger.info(f"Estimated cost: ${est_cost:.4f}")
    
    # Load existing embeddings if incremental
    embedded_chunks = []
    if incremental and EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, 'r') as f:
            existing_data = json.load(f)
            embedded_chunks = existing_data.get("chunks", [])
        logger.info(f"Loaded {len(embedded_chunks)} existing embeddings")
    
    # Process in batches
    total_batches = (len(chunks_to_process) + batch_size - 1) // batch_size
    stats = {"embedded": 0, "failed": 0, "tokens_used": 0}
    
    start_time = datetime.now()
    
    for batch_idx in range(0, len(chunks_to_process), batch_size):
        batch = chunks_to_process[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        try:
            # Prepare texts for embedding
            texts = [c.get("embedding_text", c.get("text", "")) for c in batch]
            
            # Generate embeddings
            embeddings = generate_embeddings_batch(texts, client)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = embedding
                embedded_chunks.append(chunk_with_embedding)
                progress["embedded_chunk_ids"].append(chunk["chunk_id"])
                stats["embedded"] += 1
            
            # Estimate tokens used
            batch_chars = sum(len(t) for t in texts)
            stats["tokens_used"] += batch_chars // 4
            
            logger.info(f"[Batch {batch_num}/{total_batches}] ✓ Embedded {len(batch)} chunks")
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            logger.error(f"[Batch {batch_num}/{total_batches}] ✗ Error: {e}")
            for chunk in batch:
                progress["failed_chunk_ids"].append(chunk["chunk_id"])
                stats["failed"] += 1
        
        # Save progress every 10 batches
        if batch_num % 10 == 0:
            save_progress(progress)
            
            # Save intermediate embeddings
            output_data = {
                "created_at": datetime.now().isoformat(),
                "channel_handle": CHANNEL_HANDLE,
                "channel_display_name": CHANNEL_DISPLAY_NAME,
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMENSIONS,
                "total_chunks": len(embedded_chunks),
                "chunks": embedded_chunks
            }
            with open(EMBEDDINGS_FILE, 'w') as f:
                json.dump(output_data, f)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = stats["embedded"] / elapsed * 3600 if elapsed > 0 else 0
            remaining = len(chunks_to_process) - batch_idx - len(batch)
            eta_min = remaining / (rate / 60) if rate > 0 else 0
            
            logger.info(f"--- Progress saved. Rate: {rate:.0f}/hr, ETA: {eta_min:.1f}min ---")
    
    # Final save
    save_progress(progress)
    
    output_data = {
        "created_at": datetime.now().isoformat(),
        "channel_handle": CHANNEL_HANDLE,
        "channel_display_name": CHANNEL_DISPLAY_NAME,
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "total_chunks": len(embedded_chunks),
        "total_videos": len(set(c["video_id"] for c in embedded_chunks)),
        "chunks": embedded_chunks
    }
    
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(output_data, f)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    actual_cost = (stats["tokens_used"] / 1_000_000) * 0.02
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time elapsed: {elapsed/60:.2f} minutes")
    logger.info(f"Chunks embedded: {stats['embedded']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Estimated tokens used: {stats['tokens_used']:,}")
    logger.info(f"Estimated cost: ${actual_cost:.4f}")
    logger.info(f"Total embeddings in output: {len(embedded_chunks)}")
    logger.info(f"Output file: {EMBEDDINGS_FILE}")


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate OpenAI embeddings for transcript chunks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 05_generate_embeddings_v2.py                  # Process all chunks
    python 05_generate_embeddings_v2.py --limit 100      # Process first 100 chunks
    python 05_generate_embeddings_v2.py --incremental    # Only embed new chunks
    python 05_generate_embeddings_v2.py --batch-size 50  # Custom batch size
        """
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of chunks to process (for testing)'
    )
    
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only process chunks not already embedded'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Number of chunks per API call (default: {EMBEDDING_BATCH_SIZE})'
    )
    
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    process_embeddings(
        limit=args.limit,
        incremental=args.incremental,
        batch_size=args.batch_size
    )

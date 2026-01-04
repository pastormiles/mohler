#!/usr/bin/env python3
"""
04_chunk_transcripts_v2.py
Combine small transcript segments into larger, searchable chunks with timestamps.

================================================================================
OVERVIEW
================================================================================

This script is Step 4 of the YouTube AI Semantic Search pipeline. It takes
individual transcript files (one per video) and combines their small segments
(typically 2-5 seconds each) into larger, meaningful chunks suitable for
semantic search.

Why chunking matters:
- Raw transcript segments are too small for effective search (2-5 seconds)
- Embedding models need sufficient context to understand meaning
- 60-90 second chunks capture complete thoughts/topics
- Chunks include timestamps for deep-linking to exact moments

================================================================================
HOW IT WORKS
================================================================================

1. LOAD TRANSCRIPTS
   - Reads all {video_id}.json files from data/transcripts/
   - Each file contains an array of segments with text, start time, duration

2. LOAD METADATA
   - Gets video metadata (titles, thumbnails, durations) from video_metadata.json
   - Used to enrich chunks with display information

3. CHUNK ALGORITHM
   - Iterates through segments, accumulating text until target duration reached
   - Target: 75 seconds (configurable via config.py)
   - Minimum: 45 seconds (won't create smaller chunks)
   - Maximum: 120 seconds (forces split even mid-sentence)
   - Short final segments get merged into previous chunk

4. CREATE RICH CHUNKS
   Each chunk contains:
   - chunk_id: Unique identifier (yt-{video_id}-{index})
   - text: Combined segment text
   - start_time/end_time: Seconds from video start
   - start_timestamp/end_timestamp: Human-readable (MM:SS or H:MM:SS)
   - video_title, channel, thumbnail_url: Display metadata
   - youtube_url: Deep link with timestamp (?t=XXs)
   - embedding_text: Formatted text optimized for embedding

5. SAVE OUTPUT
   - all_chunks.json: Complete array of all chunks
   - chunking_progress.json: Track which videos have been processed

================================================================================
INPUT FORMAT (from Script 03)
================================================================================

Each transcript file (data/transcripts/{video_id}.json):
{
    "video_id": "abc123",
    "title": "Video Title",
    "channel": "Channel Name",
    "duration_seconds": 1847,
    "segments": [
        {"text": "Hello everyone", "start": 0.0, "duration": 2.5},
        {"text": "welcome to today's video", "start": 2.5, "duration": 3.1},
        ...
    ]
}

================================================================================
OUTPUT FORMAT
================================================================================

data/chunks/all_chunks.json:
{
    "created_at": "2025-01-15T10:30:00",
    "total_chunks": 5432,
    "total_videos": 287,
    "chunks": [
        {
            "chunk_id": "yt-abc123-0000",
            "video_id": "abc123",
            "chunk_index": 0,
            "text": "Hello everyone welcome to today's video...",
            "start_time": 0.0,
            "end_time": 78.5,
            "start_timestamp": "0:00",
            "end_timestamp": "1:18",
            "duration_seconds": 78.5,
            "video_title": "Video Title",
            "channel": "Channel Name",
            "thumbnail_url": "https://i.ytimg.com/vi/abc123/hqdefault.jpg",
            "youtube_url": "https://www.youtube.com/watch?v=abc123&t=0s",
            "video_url": "https://www.youtube.com/watch?v=abc123",
            "embedding_text": "Video Title | 0:00\n\nHello everyone welcome..."
        },
        ...
    ]
}

================================================================================
USAGE
================================================================================

Basic usage (process all transcripts):
    python 04_chunk_transcripts_v2.py

Limit processing (for testing):
    python 04_chunk_transcripts_v2.py --limit 10

Incremental mode (only process new transcripts):
    python 04_chunk_transcripts_v2.py --incremental

Combined:
    python 04_chunk_transcripts_v2.py --incremental --limit 50

================================================================================
REQUIREMENTS
================================================================================

Prerequisites:
    - Python 3.8+
    - config.py in project root (created by yt_ai_search_setup.sh)
    - Completed Step 03 (transcript files exist)
    - Completed Step 02 (video_metadata.json exists)

No external API calls - this is local processing only.

================================================================================
CHUNKING PARAMETERS (from config.py)
================================================================================

TARGET_CHUNK_DURATION = 75   # Ideal chunk length in seconds
MIN_CHUNK_DURATION = 45      # Won't create chunks shorter than this
MAX_CHUNK_DURATION = 120     # Forces split if chunk exceeds this

These values are optimized for spoken content and semantic search.
Shorter chunks lose context; longer chunks dilute search relevance.

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
    TRANSCRIPTS_DIR,
    METADATA_FILE,
    CHUNKS_DIR,
    CHUNKS_FILE,
    CHUNKS_PROGRESS_FILE,
    LOGS_DIR,
    TARGET_CHUNK_DURATION,
    MIN_CHUNK_DURATION,
    MAX_CHUNK_DURATION,
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
            logging.FileHandler(get_log_file('04_chunk_transcripts')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_metadata():
    """Load video metadata for thumbnails and other info."""
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)
    videos_list = data.get("videos", [])
    return {video["video_id"]: video for video in videos_list}


def load_progress():
    """Load chunking progress."""
    if CHUNKS_PROGRESS_FILE.exists():
        with open(CHUNKS_PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "failed": []}


def save_progress(progress):
    """Save chunking progress."""
    with open(CHUNKS_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def format_timestamp(seconds):
    """Convert seconds to human-readable timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def create_youtube_link(video_id, start_seconds):
    """Create YouTube deep link with timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"


# =============================================================================
# CHUNKING LOGIC
# =============================================================================

def chunk_transcript(transcript_data, video_metadata):
    """
    Split transcript into semantic chunks of ~60-90 seconds.
    
    Args:
        transcript_data: Dict with video_id, title, segments, etc.
        video_metadata: Dict with thumbnail_url, description, etc.
    
    Returns:
        List of chunk dicts ready for embedding
    """
    video_id = transcript_data["video_id"]
    segments = transcript_data.get("segments", [])
    
    if not segments:
        return []
    
    chunks = []
    current_chunk_text = []
    current_chunk_start = segments[0]["start"]
    current_chunk_duration = 0
    chunk_index = 0
    
    for segment in segments:
        segment_text = segment["text"].strip()
        segment_start = segment["start"]
        segment_duration = segment.get("duration", 2.0)
        
        # Skip empty segments
        if not segment_text:
            continue
        
        # Check if adding this segment would exceed max duration
        potential_duration = (segment_start + segment_duration) - current_chunk_start
        
        # If we've reached target duration and have content, finalize chunk
        if current_chunk_duration >= TARGET_CHUNK_DURATION and current_chunk_text:
            chunk = create_chunk(
                video_id=video_id,
                chunk_index=chunk_index,
                text=current_chunk_text,
                start_time=current_chunk_start,
                end_time=segment_start,
                transcript_data=transcript_data,
                video_metadata=video_metadata
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Start new chunk
            current_chunk_text = [segment_text]
            current_chunk_start = segment_start
            current_chunk_duration = segment_duration
        
        # If adding would exceed max, force new chunk
        elif potential_duration > MAX_CHUNK_DURATION and current_chunk_text:
            chunk = create_chunk(
                video_id=video_id,
                chunk_index=chunk_index,
                text=current_chunk_text,
                start_time=current_chunk_start,
                end_time=segment_start,
                transcript_data=transcript_data,
                video_metadata=video_metadata
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Start new chunk
            current_chunk_text = [segment_text]
            current_chunk_start = segment_start
            current_chunk_duration = segment_duration
        
        else:
            # Add to current chunk
            current_chunk_text.append(segment_text)
            current_chunk_duration = (segment_start + segment_duration) - current_chunk_start
    
    # Don't forget the last chunk
    if current_chunk_text:
        last_segment = segments[-1]
        end_time = last_segment["start"] + last_segment.get("duration", 2.0)
        
        # Only create if meets minimum duration (or it's the only content)
        if current_chunk_duration >= MIN_CHUNK_DURATION or chunk_index == 0:
            chunk = create_chunk(
                video_id=video_id,
                chunk_index=chunk_index,
                text=current_chunk_text,
                start_time=current_chunk_start,
                end_time=end_time,
                transcript_data=transcript_data,
                video_metadata=video_metadata
            )
            chunks.append(chunk)
        elif chunks:
            # Merge with previous chunk if too short
            chunks[-1]["text"] += " " + " ".join(current_chunk_text)
            chunks[-1]["end_time"] = end_time
            chunks[-1]["end_timestamp"] = format_timestamp(end_time)
            chunks[-1]["duration_seconds"] = end_time - chunks[-1]["start_time"]
    
    return chunks


def create_chunk(video_id, chunk_index, text, start_time, end_time, 
                 transcript_data, video_metadata):
    """Create a chunk dict with all necessary metadata."""
    
    # Combine segment texts
    combined_text = " ".join(text)
    
    # Clean up text (remove multiple spaces, etc.)
    combined_text = " ".join(combined_text.split())
    
    return {
        "chunk_id": f"yt-{video_id}-{chunk_index:04d}",
        "video_id": video_id,
        "chunk_index": chunk_index,
        
        # Text content
        "text": combined_text,
        
        # Timestamps
        "start_time": start_time,
        "end_time": end_time,
        "start_timestamp": format_timestamp(start_time),
        "end_timestamp": format_timestamp(end_time),
        "duration_seconds": end_time - start_time,
        
        # Video metadata
        "video_title": transcript_data.get("title", ""),
        "channel": transcript_data.get("channel", CHANNEL_DISPLAY_NAME),
        "video_duration_seconds": transcript_data.get("duration_seconds", 0),
        "thumbnail_url": video_metadata.get("thumbnail_url", ""),
        
        # Links
        "youtube_url": create_youtube_link(video_id, start_time),
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        
        # For embedding context
        "embedding_text": f"{transcript_data.get('title', '')} | {format_timestamp(start_time)}\n\n{combined_text}"
    }


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_all_transcripts(limit=None, incremental=False):
    """Process all transcript files into chunks."""
    
    logger.info("=" * 60)
    logger.info("Starting Transcript Chunking")
    logger.info(f"Channel: {CHANNEL_DISPLAY_NAME} ({CHANNEL_HANDLE})")
    logger.info("=" * 60)
    
    logger.info(f"Chunking parameters:")
    logger.info(f"  Target duration: {TARGET_CHUNK_DURATION}s")
    logger.info(f"  Min duration: {MIN_CHUNK_DURATION}s")
    logger.info(f"  Max duration: {MAX_CHUNK_DURATION}s")
    
    # Load metadata for thumbnails
    metadata = load_metadata()
    progress = load_progress()
    
    # Get all transcript files
    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    transcript_files = [f for f in transcript_files if f.name != "progress.json"]
    
    if incremental:
        # Only process new transcripts
        already_processed = set(progress.get("processed", []))
        transcript_files = [f for f in transcript_files 
                          if f.stem not in already_processed]
        logger.info(f"Incremental mode: {len(already_processed)} already processed")
    
    if limit:
        transcript_files = transcript_files[:limit]
        logger.info(f"Limit mode: processing first {limit} transcripts")
    
    logger.info(f"Found {len(transcript_files)} transcript files to process")
    
    if not transcript_files:
        logger.info("No new transcripts to process!")
        return
    
    # Load existing chunks if incremental
    all_chunks = []
    if incremental and CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, 'r') as f:
            existing_data = json.load(f)
            all_chunks = existing_data.get("chunks", [])
        logger.info(f"Loaded {len(all_chunks)} existing chunks")
    
    # Process each transcript
    stats = {"processed": 0, "chunks_created": 0, "failed": 0}
    
    for i, transcript_file in enumerate(transcript_files, 1):
        video_id = transcript_file.stem
        
        try:
            # Load transcript
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Get video metadata
            video_meta = metadata.get(video_id, {})
            
            # Create chunks
            chunks = chunk_transcript(transcript_data, video_meta)
            
            if chunks:
                all_chunks.extend(chunks)
                progress["processed"].append(video_id)
                stats["processed"] += 1
                stats["chunks_created"] += len(chunks)
                
                logger.info(f"[{i}/{len(transcript_files)}] ✓ {video_id} - "
                           f"{transcript_data.get('title', '')[:40]}... "
                           f"({len(chunks)} chunks)")
            else:
                logger.warning(f"[{i}/{len(transcript_files)}] ⚠ {video_id} - No segments found")
                progress["failed"].append(video_id)
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"[{i}/{len(transcript_files)}] ✗ {video_id} - Error: {e}")
            progress["failed"].append(video_id)
            stats["failed"] += 1
        
        # Save progress every 100 files
        if i % 100 == 0:
            save_progress(progress)
            logger.info(f"--- Progress saved: {stats['chunks_created']} chunks created ---")
    
    # Save all chunks
    output_data = {
        "created_at": datetime.now().isoformat(),
        "channel_handle": CHANNEL_HANDLE,
        "channel_display_name": CHANNEL_DISPLAY_NAME,
        "total_chunks": len(all_chunks),
        "total_videos": len(set(c["video_id"] for c in all_chunks)),
        "chunking_parameters": {
            "target_duration": TARGET_CHUNK_DURATION,
            "min_duration": MIN_CHUNK_DURATION,
            "max_duration": MAX_CHUNK_DURATION,
        },
        "chunks": all_chunks
    }
    
    with open(CHUNKS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Final progress save
    save_progress(progress)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CHUNKING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Transcripts processed: {stats['processed']}")
    logger.info(f"Chunks created: {stats['chunks_created']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total chunks in output: {len(all_chunks)}")
    logger.info(f"Output file: {CHUNKS_FILE}")
    
    # Chunk stats
    if all_chunks:
        durations = [c["duration_seconds"] for c in all_chunks]
        avg_duration = sum(durations) / len(durations)
        logger.info(f"Average chunk duration: {avg_duration:.1f}s")
        logger.info(f"Min chunk duration: {min(durations):.1f}s")
        logger.info(f"Max chunk duration: {max(durations):.1f}s")


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chunk transcript segments into ~60-90 second searchable chunks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 04_chunk_transcripts_v2.py                  # Process all transcripts
    python 04_chunk_transcripts_v2.py --limit 10       # Process first 10 transcripts
    python 04_chunk_transcripts_v2.py --incremental    # Only process new transcripts
        """
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of transcripts to process (for testing)'
    )
    
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only process transcripts not already chunked'
    )
    
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    process_all_transcripts(limit=args.limit, incremental=args.incremental)

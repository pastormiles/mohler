#!/usr/bin/env python3
"""
02_fetch_video_metadata_v3.py
Fetch detailed metadata for all videos from a YouTube channel.

================================================================================
OVERVIEW
================================================================================

This script is Step 2 of the YouTube AI Semantic Search pipeline. It takes the
video IDs from Step 1 and fetches detailed metadata from the YouTube Data API,
including duration, view counts, tags, and caption availability.

Why this step matters:
- Duration helps filter out very short clips (intros, teasers)
- View counts help prioritize popular content
- Caption availability indicates transcript quality
- Tags provide additional categorization hints

================================================================================
HOW IT WORKS
================================================================================

1. LOAD VIDEO IDS
   - Reads all_video_ids.json from Step 01
   - Contains basic info: video_id, title, description, publish date

2. BATCH API REQUESTS
   - YouTube API allows up to 50 video IDs per request
   - Script batches IDs efficiently to minimize API calls
   - Small delay between batches to respect rate limits

3. FETCH DETAILED METADATA
   For each video, retrieves:
   - contentDetails: duration, caption availability
   - statistics: views, likes, comments
   - snippet: tags, full description, language

4. ENRICH DATA
   - Parse ISO 8601 duration to seconds
   - Categorize by duration (short, medium, long, etc.)
   - Estimate transcript chunk count
   - Calculate total channel duration

5. SAVE RESULTS
   - video_metadata.json: All videos with full metadata
   - Includes summary statistics for the channel

================================================================================
INPUT FORMAT (from Script 01)
================================================================================

data/video_ids/all_video_ids.json:
{
    "extraction_date": "2025-01-15T10:30:00",
    "channel_handle": "@AlbertMohler",
    "total_videos": 1547,
    "videos": [
        {
            "video_id": "abc123xyz",
            "title": "Video Title",
            "description": "...",
            "published_at": "2024-03-15T14:00:00Z",
            ...
        }
    ]
}

================================================================================
OUTPUT FORMAT
================================================================================

data/metadata/video_metadata.json:
{
    "fetch_date": "2025-01-15T11:00:00",
    "channel_handle": "@AlbertMohler",
    "channel_display_name": "Albert Mohler",
    "total_videos": 1547,
    "total_duration_seconds": 2847600,
    "total_duration_formatted": "791:00:00",
    "estimated_total_chunks": 38234,
    "duration_distribution": {
        "very_short": 23,
        "short": 145,
        "medium": 892,
        "long": 412,
        "very_long": 75
    },
    "videos": [
        {
            "video_id": "abc123xyz",
            "title": "Video Title",
            "duration_seconds": 1847,
            "duration_formatted": "30:47",
            "duration_iso": "PT30M47S",
            "duration_category": "medium",
            "view_count": 15432,
            "like_count": 523,
            "comment_count": 87,
            "tags": ["theology", "bible", "commentary"],
            "full_description": "Full video description...",
            "caption_available": true,
            "estimated_chunks": 25,
            ...
        }
    ]
}

================================================================================
USAGE
================================================================================

Basic usage (fetch all videos):
    python 02_fetch_video_metadata_v3.py

Limit processing (for testing):
    python 02_fetch_video_metadata_v3.py --limit 100

================================================================================
REQUIREMENTS
================================================================================

Prerequisites:
    - Python 3.8+
    - config.py in project root (created by yt_ai_search_setup.sh)
    - Completed Step 01 (all_video_ids.json exists)
    - YouTube Data API key in .env file

Dependencies:
    - google-api-python-client
    - tqdm (for progress bar)

API Key Setup:
    1. Go to https://console.cloud.google.com/
    2. Enable "YouTube Data API v3"
    3. Create credentials → API Key
    4. Add to .env: YOUTUBE_API_KEY=AIza...

================================================================================
API QUOTA
================================================================================

YouTube Data API has a daily quota of 10,000 units (free tier).

Cost per operation:
- videos.list: 1 unit per request (up to 50 videos each)

Example for 1,500 videos:
- 1,500 ÷ 50 = 30 requests × 1 unit = 30 units
- Well under daily limit

================================================================================
DURATION CATEGORIES
================================================================================

Videos are categorized by duration for filtering:

- very_short: < 1 minute (likely intros, clips)
- short: 1-5 minutes
- medium: 5-30 minutes
- long: 30-60 minutes
- very_long: > 1 hour

================================================================================
CHUNK ESTIMATION
================================================================================

The script estimates how many transcript chunks each video will produce:

Assumptions:
- Average speaking rate: 150 words per minute
- Target chunk size: ~300 words (60-90 seconds of speech)

Formula:
- total_words = (duration_seconds / 60) × 150
- estimated_chunks = total_words / 300

This helps estimate:
- Embedding costs (OpenAI charges per token)
- Pinecone storage requirements
- Processing time

================================================================================
CONFIG IMPORTS USED
================================================================================

from config import (
    CHANNEL_HANDLE,           # "@AlbertMohler"
    CHANNEL_DISPLAY_NAME,     # "Albert Mohler"
    YOUTUBE_API_KEY,          # From .env
    VIDEO_IDS_FILE,           # data/video_ids/all_video_ids.json
    METADATA_FILE,            # data/metadata/video_metadata.json
    LOGS_DIR,                 # logs/
    ensure_directories,
    get_log_file,
)

================================================================================
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

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
    YOUTUBE_API_KEY,
    VIDEO_IDS_FILE,
    METADATA_FILE,
    LOGS_DIR,
    ensure_directories,
    get_log_file,
)

# =============================================================================
# CONSTANTS
# =============================================================================

BATCH_SIZE = 50  # YouTube API allows up to 50 video IDs per request

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging with file and console handlers."""
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(get_log_file('02_fetch_video_metadata')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# YOUTUBE API HELPERS
# =============================================================================

def get_youtube_client():
    """Initialize YouTube API client."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "google-api-python-client not installed!\n"
            "Run: pip install google-api-python-client"
        )
    
    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY not found!\n"
            "Add to .env file: YOUTUBE_API_KEY=AIza...\n"
            "Get key from: https://console.cloud.google.com/"
        )
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)


def parse_duration(duration_str):
    """
    Parse ISO 8601 duration to seconds.
    
    Examples:
        PT1H30M45S -> 5445 seconds
        PT5M30S -> 330 seconds
        PT45S -> 45 seconds
    """
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def fetch_video_details(youtube, video_ids):
    """
    Fetch detailed metadata for a batch of video IDs.
    
    Args:
        youtube: YouTube API client
        video_ids: List of video IDs (max 50)
    
    Returns:
        Dict mapping video_id to metadata
    """
    from googleapiclient.errors import HttpError
    
    try:
        request = youtube.videos().list(
            part="contentDetails,statistics,snippet",
            id=','.join(video_ids)
        )
        response = request.execute()
        
        results = {}
        for item in response.get('items', []):
            video_id = item['id']
            
            duration_iso = item['contentDetails'].get('duration', 'PT0S')
            duration_seconds = parse_duration(duration_iso)
            
            stats = item.get('statistics', {})
            
            results[video_id] = {
                'duration_seconds': duration_seconds,
                'duration_formatted': format_duration(duration_seconds),
                'duration_iso': duration_iso,
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'tags': item['snippet'].get('tags', []),
                'full_description': item['snippet'].get('description', ''),
                'category_id': item['snippet'].get('categoryId', ''),
                'default_language': item['snippet'].get('defaultLanguage', ''),
                'default_audio_language': item['snippet'].get('defaultAudioLanguage', ''),
                'caption_available': item['contentDetails'].get('caption', 'false') == 'true'
            }
        
        return results
        
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        raise


# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_video_ids():
    """Load video IDs from extraction output."""
    if not VIDEO_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {VIDEO_IDS_FILE}\n"
            "Please run 01_extract_video_ids_v2.py first."
        )
    
    with open(VIDEO_IDS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['videos']


def categorize_by_duration(duration_seconds):
    """Categorize video by duration."""
    if duration_seconds < 60:
        return 'very_short'  # < 1 min (likely intros, clips)
    elif duration_seconds < 300:
        return 'short'  # 1-5 min
    elif duration_seconds < 1800:
        return 'medium'  # 5-30 min
    elif duration_seconds < 3600:
        return 'long'  # 30-60 min
    else:
        return 'very_long'  # > 1 hour


def estimate_transcript_chunks(duration_seconds, words_per_minute=150, chunk_words=300):
    """
    Estimate number of transcript chunks for a video.
    
    Assumes:
    - Average speaking rate of 150 words per minute
    - Target chunk size of ~300 words
    """
    total_words = (duration_seconds / 60) * words_per_minute
    estimated_chunks = max(1, int(total_words / chunk_words))
    return estimated_chunks


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fetch detailed metadata for YouTube videos.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 02_fetch_video_metadata_v3.py              # Fetch all videos
    python 02_fetch_video_metadata_v3.py --limit 100  # Fetch first 100 videos
        """
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of videos to process (default: all)'
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution flow."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Video Metadata Fetch")
    logger.info(f"Channel: {CHANNEL_DISPLAY_NAME} ({CHANNEL_HANDLE})")
    logger.info("=" * 60)
    
    if not YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY not set!")
        logger.error("Add to .env file: YOUTUBE_API_KEY=AIza...")
        logger.error("Get key from: https://console.cloud.google.com/")
        return
    
    # Load existing video data
    logger.info(f"Loading video IDs from {VIDEO_IDS_FILE}")
    videos = load_video_ids()
    logger.info(f"Loaded {len(videos)} videos")
    
    if args.limit:
        videos = videos[:args.limit]
        logger.info(f"Limit mode: processing first {args.limit} videos")
    
    # Initialize API client
    youtube = get_youtube_client()
    
    # Extract just the video IDs
    video_ids = [v['video_id'] for v in videos]
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed, progress bar disabled")
        tqdm = lambda x, **kwargs: x
    
    # Fetch metadata in batches
    logger.info(f"Fetching metadata in batches of {BATCH_SIZE}...")
    all_metadata = {}
    
    batches = [video_ids[i:i+BATCH_SIZE] for i in range(0, len(video_ids), BATCH_SIZE)]
    
    for batch in tqdm(batches, desc="Fetching metadata"):
        try:
            from googleapiclient.errors import HttpError
            metadata = fetch_video_details(youtube, batch)
            all_metadata.update(metadata)
            time.sleep(0.1)  # Small delay to be nice to the API
            
        except HttpError as e:
            if 'quotaExceeded' in str(e):
                logger.error("API quota exceeded! Try again tomorrow.")
                break
            raise
    
    logger.info(f"Fetched metadata for {len(all_metadata)} videos")
    
    # Merge metadata with original video data
    enriched_videos = []
    duration_categories = {}
    total_duration_seconds = 0
    total_estimated_chunks = 0
    
    for video in videos:
        vid = video['video_id']
        if vid in all_metadata:
            video.update(all_metadata[vid])
            
            dur_cat = categorize_by_duration(video['duration_seconds'])
            video['duration_category'] = dur_cat
            duration_categories[dur_cat] = duration_categories.get(dur_cat, 0) + 1
            
            est_chunks = estimate_transcript_chunks(video['duration_seconds'])
            video['estimated_chunks'] = est_chunks
            
            total_duration_seconds += video['duration_seconds']
            total_estimated_chunks += est_chunks
        
        enriched_videos.append(video)
    
    # Save enriched data
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'fetch_date': datetime.now().isoformat(),
        'channel_handle': CHANNEL_HANDLE,
        'channel_display_name': CHANNEL_DISPLAY_NAME,
        'total_videos': len(enriched_videos),
        'total_duration_seconds': total_duration_seconds,
        'total_duration_formatted': format_duration(total_duration_seconds),
        'estimated_total_chunks': total_estimated_chunks,
        'duration_distribution': duration_categories,
        'videos': enriched_videos
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved enriched metadata to {METADATA_FILE}")
    
    # Summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("METADATA FETCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total videos: {len(enriched_videos)}")
    logger.info(f"Total duration: {format_duration(total_duration_seconds)} ({total_duration_seconds / 3600:.1f} hours)")
    logger.info(f"Estimated total chunks: {total_estimated_chunks:,}")
    
    logger.info("")
    logger.info("Duration Distribution:")
    for cat in ['very_short', 'short', 'medium', 'long', 'very_long']:
        count = duration_categories.get(cat, 0)
        if count > 0:
            logger.info(f"  {cat}: {count} videos")
    
    with_captions = sum(1 for v in enriched_videos if v.get('caption_available', False))
    logger.info(f"\nVideos with captions: {with_captions} ({100*with_captions/len(enriched_videos):.1f}%)")
    
    logger.info("")
    logger.info("Top 10 Most Viewed Videos:")
    sorted_by_views = sorted(enriched_videos, key=lambda x: x.get('view_count', 0), reverse=True)
    for i, v in enumerate(sorted_by_views[:10], 1):
        logger.info(f"  {i}. {v['title'][:55]}... ({v.get('view_count', 0):,} views)")
    
    # Embedding cost estimate
    estimated_tokens = total_estimated_chunks * 400
    estimated_cost = (estimated_tokens / 1_000_000) * 0.02
    logger.info(f"\nEstimated embedding cost: ${estimated_cost:.2f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

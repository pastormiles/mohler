#!/usr/bin/env python3
"""
01_extract_video_ids_v3.py
Extract all video IDs from a YouTube channel.

================================================================================
OVERVIEW
================================================================================

This script is Step 1 of the YouTube AI Semantic Search pipeline. It uses the
YouTube Data API to discover all videos on a channel and save their IDs and
basic metadata for processing in subsequent steps.

Why this step matters:
- YouTube doesn't provide a simple "get all videos" endpoint
- We need video IDs to fetch transcripts, metadata, etc.
- Basic categorization helps prioritize content types

================================================================================
VERSION 3 CHANGES
================================================================================

- Added CHANNEL_ID support: If CHANNEL_ID is set in config.py, uses it directly
  instead of searching by handle. This is MORE RELIABLE because handle search
  can return wrong channels (e.g., "@pastormiles" might find a different
  Pastor Miles than intended).

- To use: Set CHANNEL_ID in config.py to your channel's UC... ID
  Find your channel ID at: youtube.com/account_advanced

================================================================================
HOW IT WORKS
================================================================================

1. RESOLVE CHANNEL ID
   - If CHANNEL_ID is set in config → use it directly (recommended)
   - If CHANNEL_ID is empty → search by handle (less reliable)
   - Channel IDs look like: UCxxxxxxxxxxxxxxxxxxxxxxxx

2. GET UPLOADS PLAYLIST
   - Every YouTube channel has a hidden "uploads" playlist
   - Contains all public videos in upload order
   - Playlist ID is derived from channel ID (UC... → UU...)

3. PAGINATE THROUGH PLAYLIST
   - YouTube API returns max 50 items per request
   - Script automatically paginates through all videos
   - Extracts video ID, title, description, publish date, thumbnail

4. CATEGORIZE VIDEOS
   - Attempts to classify videos by content type
   - Categories: bible_teaching, qa_session, sermon, special, unknown
   - Based on title/description pattern matching

5. SAVE RESULTS
   - all_video_ids.json: Full data with metadata
   - video_ids_only.txt: Simple list for quick reference

================================================================================
OUTPUT FORMAT
================================================================================

data/video_ids/all_video_ids.json:
{
    "extraction_date": "2025-01-15T10:30:00",
    "channel_handle": "@AlbertMohler",
    "channel_id": "UCxxxxxxxxxxxxxx",
    "channel_display_name": "Albert Mohler",
    "total_videos": 1547,
    "videos": [
        {
            "video_id": "abc123xyz",
            "title": "Video Title Here",
            "description": "First 500 chars of description...",
            "published_at": "2024-03-15T14:00:00Z",
            "thumbnail_url": "https://i.ytimg.com/vi/abc123xyz/maxresdefault.jpg",
            "channel_title": "Albert Mohler",
            "playlist_position": 0,
            "category": "bible_teaching",
            "url": "https://www.youtube.com/watch?v=abc123xyz"
        },
        ...
    ]
}

data/video_ids/video_ids_only.txt:
abc123xyz
def456uvw
ghi789rst
...

================================================================================
USAGE
================================================================================

Basic usage (extract all videos):
    python 01_extract_video_ids_v3.py

Limit extraction (for testing):
    python 01_extract_video_ids_v3.py --limit 50

================================================================================
REQUIREMENTS
================================================================================

Prerequisites:
    - Python 3.8+
    - config.py in project root (created by yt_ai_search_setup.sh)
    - YouTube Data API key in .env file

Dependencies:
    - google-api-python-client

API Key Setup:
    1. Go to https://console.cloud.google.com/
    2. Create a project (or select existing)
    3. Enable "YouTube Data API v3"
    4. Create credentials → API Key
    5. Add to .env: YOUTUBE_API_KEY=AIza...

================================================================================
API QUOTA
================================================================================

YouTube Data API has a daily quota of 10,000 units (free tier).

Cost per operation:
- search.list: 100 units (only used if CHANNEL_ID not set)
- channels.list: 1 unit (used once to get uploads playlist)
- playlistItems.list: 1 unit per request (50 videos each)

Example for 1,500 videos WITH CHANNEL_ID set:
- Playlist ID: 1 unit
- Video extraction: 30 requests × 1 unit = 30 units
- Total: ~31 units (much cheaper than handle search!)

Example for 1,500 videos WITHOUT CHANNEL_ID:
- Channel lookup: 100 units
- Playlist ID: 1 unit
- Video extraction: 30 requests × 1 unit = 30 units
- Total: ~131 units

================================================================================
CATEGORIZATION LOGIC
================================================================================

The script attempts to categorize videos based on title/description patterns:

- bible_teaching: Contains book names (Genesis, Matthew, etc.)
- qa_session: Contains "Q&A", "questions", etc.
- sermon: Contains "sermon", "Sunday", "church"
- special: Interviews, special topics
- unknown: Cannot determine

This is a rough heuristic - actual categories may need manual review.

================================================================================
CONFIG IMPORTS USED
================================================================================

from config import (
    CHANNEL_HANDLE,           # "@AlbertMohler"
    CHANNEL_ID,               # "UCxxxxxx" (optional but recommended)
    CHANNEL_DISPLAY_NAME,     # "Albert Mohler"
    CHANNEL_URL,              # "https://www.youtube.com/channel/UCxxxxxx"
    YOUTUBE_API_KEY,          # From .env
    VIDEO_IDS_DIR,            # data/video_ids/
    VIDEO_IDS_FILE,           # data/video_ids/all_video_ids.json
    VIDEO_IDS_ONLY_FILE,      # data/video_ids/video_ids_only.txt
    LOGS_DIR,                 # logs/
    ensure_directories,
    get_log_file,
)

================================================================================
"""

import argparse
import json
import logging
import sys
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
    CHANNEL_ID,
    CHANNEL_DISPLAY_NAME,
    CHANNEL_URL,
    YOUTUBE_API_KEY,
    VIDEO_IDS_DIR,
    VIDEO_IDS_FILE,
    VIDEO_IDS_ONLY_FILE,
    LOGS_DIR,
    ensure_directories,
    get_log_file,
)

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
            logging.FileHandler(get_log_file('01_extract_video_ids')),
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


def get_channel_id(youtube, handle):
    """
    Get channel ID from channel handle.
    
    The handle is the @username format (e.g., @AlbertMohler)
    """
    from googleapiclient.errors import HttpError
    
    logger.info(f"Looking up channel ID for handle: {handle}")
    
    # Remove @ if present
    handle_clean = handle.lstrip('@')
    
    try:
        request = youtube.search().list(
            part="snippet",
            q=handle_clean,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if response['items']:
            channel_id = response['items'][0]['snippet']['channelId']
            channel_title = response['items'][0]['snippet']['title']
            logger.info(f"Found channel: {channel_title} (ID: {channel_id})")
            return channel_id
        else:
            raise ValueError(f"No channel found for handle: {handle}")
            
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        raise


def get_uploads_playlist_id(youtube, channel_id):
    """
    Get the 'uploads' playlist ID for a channel.
    
    Every YouTube channel has a hidden 'uploads' playlist containing all their videos.
    The playlist ID is derived from the channel ID by replacing 'UC' with 'UU'.
    """
    from googleapiclient.errors import HttpError
    
    logger.info(f"Getting uploads playlist for channel: {channel_id}")
    
    try:
        request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()
        
        if response['items']:
            uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            logger.info(f"Uploads playlist ID: {uploads_playlist_id}")
            return uploads_playlist_id
        else:
            raise ValueError(f"No channel found with ID: {channel_id}")
            
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        raise


def get_all_video_ids(youtube, playlist_id, limit=None):
    """
    Get all video IDs from a playlist with pagination.
    
    YouTube API returns max 50 items per request, so we need to paginate.
    
    Args:
        youtube: YouTube API client
        playlist_id: The uploads playlist ID
        limit: Optional maximum number of videos to retrieve
    """
    from googleapiclient.errors import HttpError
    
    logger.info(f"Extracting video IDs from playlist: {playlist_id}")
    if limit:
        logger.info(f"Limit mode: extracting first {limit} videos")
    
    videos = []
    next_page_token = None
    page_count = 0
    
    while True:
        try:
            request = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=50,  # Maximum allowed
                pageToken=next_page_token
            )
            response = request.execute()
            
            page_count += 1
            
            for item in response['items']:
                video_data = {
                    'video_id': item['contentDetails']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', '')[:500],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': get_best_thumbnail(item['snippet'].get('thumbnails', {})),
                    'channel_title': item['snippet']['channelTitle'],
                    'playlist_position': item['snippet']['position']
                }
                videos.append(video_data)
                
                if limit and len(videos) >= limit:
                    logger.info(f"Reached limit of {limit} videos")
                    return videos
            
            logger.info(f"Page {page_count}: Retrieved {len(response['items'])} videos (Total: {len(videos)})")
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        except HttpError as e:
            logger.error(f"YouTube API error on page {page_count}: {e}")
            raise
    
    logger.info(f"✓ Completed: {len(videos)} total videos extracted")
    return videos


def get_best_thumbnail(thumbnails):
    """Get the best quality thumbnail URL available."""
    for quality in ['maxres', 'standard', 'high', 'medium', 'default']:
        if quality in thumbnails:
            return thumbnails[quality]['url']
    return None


# =============================================================================
# VIDEO CATEGORIZATION
# =============================================================================

def categorize_video(title, description):
    """
    Attempt to categorize video based on title/description.
    
    Categories:
    - bible_teaching: Verse-by-verse commentary
    - qa_session: Q&A format
    - sermon: Church sermons
    - special: Special topics, interviews, etc.
    - unknown: Cannot determine
    """
    title_lower = title.lower()
    desc_lower = description.lower()
    
    bible_books = [
        'genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy',
        'joshua', 'judges', 'ruth', 'samuel', 'kings', 'chronicles',
        'ezra', 'nehemiah', 'esther', 'job', 'psalm', 'proverbs',
        'ecclesiastes', 'song of solomon', 'isaiah', 'jeremiah',
        'lamentations', 'ezekiel', 'daniel', 'hosea', 'joel', 'amos',
        'obadiah', 'jonah', 'micah', 'nahum', 'habakkuk', 'zephaniah',
        'haggai', 'zechariah', 'malachi', 'matthew', 'mark', 'luke',
        'john', 'acts', 'romans', 'corinthians', 'galatians', 'ephesians',
        'philippians', 'colossians', 'thessalonians', 'timothy', 'titus',
        'philemon', 'hebrews', 'james', 'peter', 'jude', 'revelation'
    ]
    
    if 'q&a' in title_lower or 'q & a' in title_lower or 'questions' in title_lower:
        return 'qa_session'
    
    for book in bible_books:
        if book in title_lower:
            return 'bible_teaching'
    
    if any(book in desc_lower for book in bible_books):
        return 'bible_teaching'
    
    if 'sermon' in title_lower or 'sunday' in title_lower or 'church' in title_lower:
        return 'sermon'
    
    return 'unknown'


def enrich_video_data(videos):
    """Add derived fields to video data."""
    logger.info("Enriching video data with categories...")
    
    category_counts = {}
    
    for video in videos:
        category = categorize_video(video['title'], video['description'])
        video['category'] = category
        video['url'] = f"https://www.youtube.com/watch?v={video['video_id']}"
        category_counts[category] = category_counts.get(category, 0) + 1
    
    logger.info(f"Category distribution: {category_counts}")
    return videos


# =============================================================================
# OUTPUT
# =============================================================================

def save_results(videos, channel_id):
    """Save video data to JSON file."""
    VIDEO_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'extraction_date': datetime.now().isoformat(),
        'channel_handle': CHANNEL_HANDLE,
        'channel_id': channel_id,
        'channel_display_name': CHANNEL_DISPLAY_NAME,
        'total_videos': len(videos),
        'videos': videos
    }
    
    with open(VIDEO_IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(videos)} videos to {VIDEO_IDS_FILE}")
    
    with open(VIDEO_IDS_ONLY_FILE, 'w') as f:
        for video in videos:
            f.write(f"{video['video_id']}\n")
    
    logger.info(f"✓ Saved video IDs list to {VIDEO_IDS_ONLY_FILE}")


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract all video IDs from a YouTube channel.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 01_extract_video_ids_v3.py              # Extract all videos
    python 01_extract_video_ids_v3.py --limit 50   # Extract first 50 videos
        """
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of videos to extract (default: all)'
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution flow."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting YouTube Video ID Extraction")
    logger.info(f"Channel: {CHANNEL_DISPLAY_NAME} ({CHANNEL_HANDLE})")
    if CHANNEL_ID:
        logger.info(f"Channel ID: {CHANNEL_ID} (from config)")
    logger.info(f"URL: {CHANNEL_URL}")
    logger.info("=" * 60)
    
    if not YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY not set!")
        logger.error("Add to .env file: YOUTUBE_API_KEY=AIza...")
        logger.error("Get key from: https://console.cloud.google.com/")
        return
    
    try:
        youtube = get_youtube_client()
        
        # Use CHANNEL_ID directly if available, otherwise search by handle
        if CHANNEL_ID:
            logger.info(f"Using CHANNEL_ID from config: {CHANNEL_ID}")
            channel_id = CHANNEL_ID
        else:
            logger.warning("CHANNEL_ID not set in config - searching by handle (less reliable)")
            logger.warning("Tip: Set CHANNEL_ID in config.py for more reliable results")
            channel_id = get_channel_id(youtube, CHANNEL_HANDLE)
        
        uploads_playlist_id = get_uploads_playlist_id(youtube, channel_id)
        videos = get_all_video_ids(youtube, uploads_playlist_id, limit=args.limit)
        videos = enrich_video_data(videos)
        save_results(videos, channel_id)
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total videos: {len(videos)}")
        logger.info(f"Output file: {VIDEO_IDS_FILE}")
        
        categories = {}
        for v in videos:
            cat = v['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info("")
        logger.info("Category Breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count} videos")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

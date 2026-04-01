# WordPress Semantic Search Extension

## Project Overview

Extend the existing Albert Mohler YouTube semantic search system to include all content from albertmohler.com. The result is a unified search experience where users enter a single query and receive results from both YouTube videos and website content, displayed in a segmented layout.

---

## Current Architecture (YouTube Search)

### Tech Stack
| Component | Technology |
|-----------|------------|
| Embeddings | OpenAI `text-embedding-3-small` (1536 dimensions) |
| Vector Database | Pinecone (index: `mohler-youtube`, namespace: `youtube`) |
| API Server | Flask with Flask-CORS |
| Caching | LRU cache (1000 entries, 1-hour TTL) |
| Analytics | SQLite |
| Automation | GitHub Actions (daily at 2 AM Central) |

### Existing Pipeline Scripts
```
scripts/
├── 01_extract_video_ids_v3.py      # Get video IDs from channel
├── 02_fetch_video_metadata_v3.py   # Fetch titles, thumbnails, durations
├── 03_extract_transcripts_v10.py   # Extract transcripts (with proxies)
├── 04_chunk_transcripts_v2.py      # 60-90 second chunks
├── 05_generate_embeddings_v2.py    # OpenAI embeddings
├── 06_upload_to_pinecone_v2.py     # Upload to Pinecone
├── 07_local_POC_v1.py              # Generate Flask server
└── 08_build_wp_plugin_v1.py        # Generate WordPress plugin
```

### Data Flow
```
YouTube Channel
    ↓
Video IDs → Metadata → Transcripts → Chunks → Embeddings → Pinecone
                                                              ↓
                                                         Flask API
                                                              ↓
                                                          Web UI
```

---

## Proposed Extension: WordPress Content

### Architecture Changes

**Same Index, New Namespace**
- Keep using Pinecone index `mohler-youtube`
- Add new namespace `website` for WordPress content
- Enables unified search across both sources with single query

**Same Embedding Model**
- Continue using `text-embedding-3-small` for consistency
- 1536-dimensional vectors
- Cost: ~$0.02 per 1M tokens

### New Pipeline Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `09_extract_wp_content_v1.py` | Fetch posts via WP REST API | WordPress site | `data/wp_content/*.json` |
| `10_chunk_wp_content_v1.py` | Semantic chunking | Raw content | `data/wp_chunks/all_wp_chunks.json` |
| `11_generate_wp_embeddings_v1.py` | OpenAI embeddings | Chunks | `data/wp_embeddings/wp_embeddings.json` |
| `12_upload_wp_to_pinecone_v1.py` | Upload to Pinecone | Embeddings | Pinecone `website` namespace |

### New Data Directories
```
data/
├── video_ids/          # Existing - YouTube
├── metadata/           # Existing - YouTube
├── transcripts/        # Existing - YouTube
├── chunks/             # Existing - YouTube
├── embeddings/         # Existing - YouTube
├── wp_content/         # NEW - Raw WordPress JSON
│   ├── progress.json   # Track extraction progress
│   └── {post_id}.json  # Individual post data
├── wp_chunks/          # NEW - Chunked content
│   └── all_wp_chunks.json
└── wp_embeddings/      # NEW - WordPress embeddings
    └── wp_embeddings.json
```

---

## WordPress Content Extraction

### Method: REST API

WordPress exposes content via the WP REST API (available in WordPress 4.7+). This is the recommended approach because:

1. No plugin installation required
2. Clean JSON output matches existing pipeline
3. Supports incremental updates via `modified_after` parameter
4. Handles all post types (posts, pages, custom)
5. Includes featured images via `_embed` parameter

### REST API Endpoints

```
Base URL: https://albertmohler.com/wp-json/wp/v2/

GET /posts              # Blog posts/articles
GET /pages              # Static pages
GET /types              # Discover all registered post types
GET /{custom_type}      # Custom post types (briefing, podcast, etc.)
GET /categories         # Category taxonomy
GET /tags               # Tag taxonomy
```

### Authentication

For public content, no authentication is required. However, for higher rate limits and access to draft/private content:

1. Go to WordPress Admin → Users → Your Profile
2. Scroll to "Application Passwords"
3. Create new application password
4. Use HTTP Basic Auth: `username:app_password`

### Example API Request

```bash
# Fetch published posts with embedded media
curl "https://albertmohler.com/wp-json/wp/v2/posts?per_page=100&page=1&_embed&status=publish"

# Discover all post types
curl "https://albertmohler.com/wp-json/wp/v2/types"

# Incremental update (posts modified after date)
curl "https://albertmohler.com/wp-json/wp/v2/posts?modified_after=2025-01-01T00:00:00&per_page=100"
```

### Expected Content Types on albertmohler.com

| Content Type | Description | Likely Endpoint |
|--------------|-------------|-----------------|
| Blog Posts | Written articles and commentary | `/wp/v2/posts` |
| The Briefing | Daily podcast with transcript | `/wp/v2/briefing` (custom) |
| Thinking in Public | Interview podcast | `/wp/v2/podcast` (custom) |
| Ask Anything | Q&A podcast | `/wp/v2/podcast` (custom) |
| Pages | Static content | `/wp/v2/pages` |

**Discovery Required**: Run `GET /wp-json/wp/v2/types` to confirm actual post type slugs.

---

## Chunking Strategy

### Why Different from YouTube

| Aspect | YouTube | WordPress |
|--------|---------|-----------|
| Natural boundary | Time (seconds) | Paragraphs/sections |
| Deep linking | Timestamp URL param | Anchor/post URL |
| Content length | Fixed (video duration) | Variable (500-10,000+ words) |
| Context needs | 60-90 seconds | 400-600 words |

### WordPress Chunking Parameters

```python
WP_TARGET_CHUNK_WORDS = 500      # Target chunk size
WP_MIN_CHUNK_WORDS = 200         # Minimum (avoid tiny chunks)
WP_MAX_CHUNK_WORDS = 800         # Maximum (force split)
WP_CHUNK_OVERLAP_WORDS = 50      # Overlap for context continuity
```

### Chunking Algorithm

```
1. Clean HTML content (strip tags, normalize whitespace)
2. Split by paragraphs (<p>, \n\n)
3. Accumulate paragraphs until target word count reached
4. Add 50-word overlap from previous chunk
5. Handle edge cases:
   - Very short posts (<300 words) → single chunk
   - Very long paragraphs → split at sentence boundaries
   - Podcast transcripts with timestamps → detect and handle
```

### Chunk Metadata Schema

```json
{
  "chunk_id": "wp-12345-0002",
  "post_id": 12345,
  "chunk_index": 2,

  "text": "The actual content text for this chunk...",
  "embedding_text": "Article Title | 2025-01-15\n\nThe actual content text...",

  "title": "The Challenge of Artificial Intelligence",
  "url": "https://albertmohler.com/2025/01/15/challenge-artificial-intelligence",
  "featured_image_url": "https://albertmohler.com/wp-content/uploads/...",

  "published_date": "2025-01-15",
  "modified_date": "2025-01-16",

  "content_type": "briefing",
  "categories": ["Culture", "Technology"],
  "author": "Albert Mohler",

  "excerpt": "First 150 characters of the post...",
  "word_count": 487
}
```

---

## API Modifications

### Updated Search Response Structure

**Current** (YouTube only):
```json
{
  "query": "artificial intelligence",
  "results": [...],
  "count": 15
}
```

**Proposed** (YouTube + Website):
```json
{
  "query": "artificial intelligence",
  "youtube": {
    "results": [
      {
        "score": 0.89,
        "chunk_id": "yt-abc123-0005",
        "video_id": "abc123",
        "video_title": "The Briefing 01-15-2025",
        "text": "Transcript excerpt...",
        "start_timestamp": "12:45",
        "end_timestamp": "14:30",
        "thumbnail_url": "https://i.ytimg.com/vi/abc123/mqdefault.jpg",
        "youtube_url": "https://www.youtube.com/watch?v=abc123&t=765s"
      }
    ],
    "count": 15,
    "unique_videos": 8
  },
  "website": {
    "results": [
      {
        "score": 0.85,
        "chunk_id": "wp-12345-0002",
        "post_id": 12345,
        "title": "The Challenge of Artificial Intelligence",
        "text": "Article excerpt...",
        "url": "https://albertmohler.com/2025/01/15/...",
        "published_date": "2025-01-15",
        "featured_image_url": "https://...",
        "content_type": "briefing",
        "categories": ["Culture", "Technology"]
      }
    ],
    "count": 12,
    "unique_posts": 6
  },
  "cached": false
}
```

### Search Endpoint Changes

```python
# server/app.py - Modified search function

@api_v1.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    top_k = request.json.get('top_k', 20)

    # Generate query embedding (same as before)
    query_embedding = generate_embedding(query)

    # Search BOTH namespaces
    youtube_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="youtube"
    )

    website_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="website"
    )

    return jsonify({
        'query': query,
        'youtube': format_youtube_results(youtube_results),
        'website': format_website_results(website_results),
        'cached': False
    })
```

### Summarize Endpoint Changes

The AI summarize feature should incorporate both sources:

```python
@api_v1.route('/summarize', methods=['POST'])
def summarize_results():
    youtube_results = request.json.get('youtube_results', [])
    website_results = request.json.get('website_results', [])

    # Build context from both sources
    context = ""

    for r in youtube_results[:3]:
        context += f"[Video: {r['video_title']} at {r['start_timestamp']}]\n"
        context += f"{r['text'][:400]}\n\n"

    for r in website_results[:3]:
        context += f"[Article: {r['title']} ({r['published_date']})]\n"
        context += f"{r['text'][:400]}\n\n"

    # Generate summary with GPT-4
    summary = generate_summary(query, context)

    return jsonify({'summary': summary})
```

---

## Frontend UI Design

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  ALBERT MOHLER SEARCH                                       │
│  ════════════════════                                       │
│  [Search Box________________________] [Search]              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AI SUMMARY                                                 │
│  ─────────                                                  │
│  Dr. Mohler has addressed artificial intelligence in        │
│  several contexts, discussing both theological and          │
│  cultural implications...                                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ▶ YOUTUBE VIDEOS                                          │
│  ─────────────────                                          │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │ thumb  │ │ thumb  │ │ thumb  │ │ thumb  │ │ thumb  │   │
│  │   1    │ │   2    │ │   3    │ │   4    │ │   5    │   │
│  │ 12:45  │ │ 5:30   │ │ 23:15  │ │ 8:00   │ │ 45:20  │   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
│                                                             │
│  Found 15 segments from 8 videos                           │
│                                                             │
│  [Expand for detailed results ▼]                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📝 WEBSITE CONTENT                                        │
│  ──────────────────                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [IMG] The Challenge of Artificial Intelligence      │   │
│  │       January 15, 2025  •  The Briefing             │   │
│  │       The rise of artificial intelligence presents  │   │
│  │       profound questions for Christian thinking...  │   │
│  │                              [Read on Website →]    │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [IMG] AI and the Image of God                       │   │
│  │       January 10, 2025  •  Culture                  │   │
│  │       What does it mean to be human in an age of    │   │
│  │       increasingly capable machines...              │   │
│  │                              [Read on Website →]    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Found 12 segments from 6 articles                         │
│                                                             │
│  [Load More]                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### CSS Components to Add

```css
/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 25px 0 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
}

.section-header h2 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Website result card */
.website-result-item {
    display: flex;
    gap: 15px;
    background: var(--bg-white);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}

.website-result-item:hover {
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}

.website-result-item .result-image {
    width: 120px;
    height: 80px;
    flex-shrink: 0;
    border-radius: 6px;
    overflow: hidden;
}

.website-result-item .result-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.result-meta {
    display: flex;
    gap: 10px;
    font-size: 0.85rem;
    color: var(--text-gray);
    margin: 5px 0 10px;
}

.result-category {
    background: var(--bg-light);
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: capitalize;
}
```

---

## GitHub Actions Automation

### Existing YouTube Automation (Already Running)

The YouTube pipeline is already automated via GitHub Actions at `.github/workflows/daily-update.yml`:

**Schedule**: Daily at 2:00 AM Central Time (8:00 AM UTC)

**Current YouTube Flow**:
```
┌─────────────────────────────────────────────────────────────────┐
│  DAILY YOUTUBE UPDATE (Already Implemented)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Download previous data from GitHub artifacts                │
│     └─ Restores: video_ids, metadata, transcripts, chunks,     │
│        embeddings from last successful run                      │
│                                                                 │
│  2. Check for new videos (--incremental)                        │
│     └─ Compares channel videos against processed list           │
│     └─ If no new videos → skip remaining steps                  │
│                                                                 │
│  3. Fetch metadata for new videos only                          │
│                                                                 │
│  4. Extract transcripts (via Webshare proxies)                  │
│                                                                 │
│  5. Chunk new transcripts (60-90 second segments)               │
│                                                                 │
│  6. Generate embeddings for new chunks                          │
│                                                                 │
│  7. Upload new vectors to Pinecone "youtube" namespace          │
│                                                                 │
│  8. Save updated data as artifact for next run                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- `--incremental` flag on each script = only processes NEW content
- GitHub artifacts persist data between runs (30-day retention)
- Conditional execution: steps 3-7 only run if new videos exist
- 2-hour timeout for large batches
- Manual trigger available via `workflow_dispatch`

### Extended Workflow: Adding WordPress

We'll add a second job to the same workflow file:

```yaml
# .github/workflows/daily-update.yml

name: Daily Content Update

on:
  schedule:
    - cron: '0 8 * * *'  # 2 AM Central Time (8 AM UTC)
  workflow_dispatch:     # Manual trigger

jobs:
  # ═══════════════════════════════════════════════════════════════
  # YOUTUBE JOB (Already exists - no changes needed)
  # ═══════════════════════════════════════════════════════════════
  update-youtube:
    runs-on: ubuntu-latest
    timeout-minutes: 120

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Create data directories
        run: mkdir -p data/video_ids data/metadata data/transcripts data/chunks data/embeddings logs

      - name: Download existing YouTube data
        uses: dawidd6/action-download-artifact@v3
        with:
          name: pipeline-data
          path: data/
          if_no_artifact_found: warn
        continue-on-error: true

      - name: Check for new videos
        id: check-new
        env:
          YOUTUBE_API_KEY: ${{ secrets.YOUTUBE_API_KEY }}
        run: |
          cd scripts
          python 01_extract_video_ids_v3.py --incremental
          NEW_COUNT=$(python -c "
          import json
          from pathlib import Path
          ids_file = Path('../data/video_ids/video_ids.json')
          if ids_file.exists():
              data = json.load(open(ids_file))
              print(len(data.get('video_ids', [])))
          else:
              print(0)
          ")
          echo "new_videos=$NEW_COUNT" >> $GITHUB_OUTPUT

      - name: Fetch metadata for new videos
        if: steps.check-new.outputs.new_videos != '0'
        env:
          YOUTUBE_API_KEY: ${{ secrets.YOUTUBE_API_KEY }}
        run: cd scripts && python 02_fetch_video_metadata_v3.py --incremental

      - name: Extract transcripts
        if: steps.check-new.outputs.new_videos != '0'
        env:
          WEBSHARE_PROXY_USERNAME: ${{ secrets.WEBSHARE_PROXY_USERNAME }}
          WEBSHARE_PROXY_PASSWORD: ${{ secrets.WEBSHARE_PROXY_PASSWORD }}
        run: cd scripts && python 03_extract_transcripts_v10.py --incremental

      - name: Chunk transcripts
        if: steps.check-new.outputs.new_videos != '0'
        run: cd scripts && python 04_chunk_transcripts_v2.py --incremental

      - name: Generate embeddings
        if: steps.check-new.outputs.new_videos != '0'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: cd scripts && python 05_generate_embeddings_v2.py --incremental

      - name: Upload to Pinecone
        if: steps.check-new.outputs.new_videos != '0'
        env:
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
        run: cd scripts && python 06_upload_to_pinecone_v2.py --incremental

      - name: Save YouTube pipeline data
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-data
          path: |
            data/video_ids/
            data/metadata/
            data/transcripts/*.json
            data/chunks/
            data/embeddings/
          retention-days: 30

  # ═══════════════════════════════════════════════════════════════
  # WORDPRESS JOB (New - to be added)
  # ═══════════════════════════════════════════════════════════════
  update-wordpress:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    # Run in parallel with YouTube (remove 'needs' line)
    # Or run after YouTube completes (keep 'needs' line)
    # needs: update-youtube

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Create WordPress data directories
        run: mkdir -p data/wp_content data/wp_chunks data/wp_embeddings logs

      - name: Download existing WordPress data
        uses: dawidd6/action-download-artifact@v3
        with:
          name: wordpress-pipeline-data
          path: data/
          if_no_artifact_found: warn
        continue-on-error: true

      - name: Check for new/modified posts
        id: check-wp
        env:
          WP_SITE_URL: ${{ secrets.WP_SITE_URL }}
          WP_APP_USER: ${{ secrets.WP_APP_USER }}
          WP_APP_PASSWORD: ${{ secrets.WP_APP_PASSWORD }}
        run: |
          cd scripts
          python 09_extract_wp_content_v1.py --incremental

          # Count new posts
          NEW_COUNT=$(python -c "
          import json
          from pathlib import Path
          progress = Path('../data/wp_content/progress.json')
          if progress.exists():
              data = json.load(open(progress))
              print(len(data.get('new_posts', [])))
          else:
              print(0)
          ")
          echo "new_posts=$NEW_COUNT" >> $GITHUB_OUTPUT

      - name: Chunk WordPress content
        if: steps.check-wp.outputs.new_posts != '0'
        run: cd scripts && python 10_chunk_wp_content_v1.py --incremental

      - name: Generate WordPress embeddings
        if: steps.check-wp.outputs.new_posts != '0'
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: cd scripts && python 11_generate_wp_embeddings_v1.py --incremental

      - name: Upload WordPress to Pinecone
        if: steps.check-wp.outputs.new_posts != '0'
        env:
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
        run: cd scripts && python 12_upload_wp_to_pinecone_v1.py --incremental

      - name: Save WordPress pipeline data
        uses: actions/upload-artifact@v4
        with:
          name: wordpress-pipeline-data
          path: |
            data/wp_content/
            data/wp_chunks/
            data/wp_embeddings/
          retention-days: 30

      - name: Generate WordPress summary
        run: |
          python -c "
          import json
          from pathlib import Path

          progress = Path('data/wp_content/progress.json')
          chunks_file = Path('data/wp_chunks/all_wp_chunks.json')

          stats = {'posts': 0, 'chunks': 0}

          if progress.exists():
              data = json.load(open(progress))
              stats['posts'] = len(data.get('processed_posts', []))

          if chunks_file.exists():
              data = json.load(open(chunks_file))
              stats['chunks'] = len(data.get('chunks', []))

          print(f\"Total posts indexed: {stats['posts']}\")
          print(f\"Total chunks: {stats['chunks']}\")
          "

      - name: Notify on failure
        if: failure()
        run: echo "::error::WordPress update pipeline failed. Check logs for details."
```

### WordPress Incremental Sync Strategy

The WordPress pipeline uses `modified_after` to detect changes:

```
┌─────────────────────────────────────────────────────────────────┐
│  WORDPRESS INCREMENTAL SYNC                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  progress.json tracks:                                          │
│  {                                                              │
│    "last_sync": "2025-01-07T08:00:00Z",                        │
│    "processed_posts": [12345, 12346, 12347, ...],              │
│    "new_posts": [],         // Populated during current run     │
│    "failed_posts": []       // For retry on next run            │
│  }                                                              │
│                                                                 │
│  API Query:                                                     │
│  GET /wp/v2/posts?modified_after=2025-01-07T08:00:00Z          │
│                                                                 │
│  Returns: Only posts created OR modified since last sync        │
│                                                                 │
│  Benefits:                                                      │
│  • New posts are indexed within 24 hours                        │
│  • Updated/edited posts are re-indexed automatically            │
│  • Minimal API calls (only fetches changes)                     │
│  • Minimal embedding cost (only new/changed chunks)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Handling Updated Posts

When a post is modified:
1. **Detection**: `modified_after` query catches the update
2. **Re-extraction**: Fresh content fetched from WordPress
3. **Re-chunking**: Content re-chunked (may produce different chunks)
4. **Pinecone update**: Old vectors deleted, new vectors uploaded
   - Delete: `index.delete(filter={"post_id": 12345}, namespace="website")`
   - Insert: Upload new chunks with same post_id

### Required GitHub Secrets

| Secret | Description | Status |
|--------|-------------|--------|
| `YOUTUBE_API_KEY` | YouTube Data API key | Already exists |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Already exists |
| `PINECONE_API_KEY` | Pinecone API key | Already exists |
| `WEBSHARE_PROXY_USERNAME` | Proxy service for YouTube | Already exists |
| `WEBSHARE_PROXY_PASSWORD` | Proxy service for YouTube | Already exists |
| `WP_SITE_URL` | `https://albertmohler.com` | **To add** |
| `WP_APP_USER` | WordPress username | **To add** |
| `WP_APP_PASSWORD` | WordPress application password | **To add** |

### Workflow Execution Options

**Option A: Sequential** (Recommended for simplicity)
```yaml
update-wordpress:
  needs: update-youtube  # WordPress waits for YouTube to finish
```
- Pros: Simpler debugging, lower resource usage
- Cons: Longer total runtime

**Option B: Parallel** (Faster)
```yaml
update-wordpress:
  # No 'needs' line = runs simultaneously with YouTube
```
- Pros: Faster total runtime (~50% time savings)
- Cons: Higher concurrent resource usage

### Manual Trigger

Both jobs can be manually triggered from GitHub:
1. Go to repository → Actions → "Daily Content Update"
2. Click "Run workflow"
3. Select branch and run

Useful for:
- Testing after code changes
- Forcing immediate sync after major content updates
- Re-running after a failed job

---

## Configuration Updates

### Additions to config.py

```python
# =============================================================================
# WORDPRESS CONFIGURATION
# =============================================================================

# Site URL
WP_SITE_URL = os.getenv("WP_SITE_URL", "https://albertmohler.com")
WP_APP_USER = os.getenv("WP_APP_USER", "")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD", "")

# API Settings
WP_POSTS_PER_PAGE = 100  # Max allowed by WP REST API
WP_REQUEST_DELAY = 0.5   # Seconds between requests (rate limiting)

# =============================================================================
# PINECONE NAMESPACES
# =============================================================================

PINECONE_NAMESPACE_YOUTUBE = "youtube"
PINECONE_NAMESPACE_WEBSITE = "website"

# =============================================================================
# WORDPRESS DATA DIRECTORIES
# =============================================================================

WP_CONTENT_DIR = DATA_DIR / "wp_content"
WP_CHUNKS_DIR = DATA_DIR / "wp_chunks"
WP_EMBEDDINGS_DIR = DATA_DIR / "wp_embeddings"

WP_CONTENT_PROGRESS_FILE = WP_CONTENT_DIR / "progress.json"
WP_CHUNKS_FILE = WP_CHUNKS_DIR / "all_wp_chunks.json"
WP_EMBEDDINGS_FILE = WP_EMBEDDINGS_DIR / "wp_embeddings.json"

# =============================================================================
# WORDPRESS CHUNKING PARAMETERS
# =============================================================================

WP_TARGET_CHUNK_WORDS = 500
WP_MIN_CHUNK_WORDS = 200
WP_MAX_CHUNK_WORDS = 800
WP_CHUNK_OVERLAP_WORDS = 50
```

---

## Implementation Checklist

### Prerequisites
- [ ] Obtain WordPress admin access to albertmohler.com
- [ ] Create WordPress Application Password
- [ ] Test REST API access: `curl https://albertmohler.com/wp-json/wp/v2/posts?per_page=1`
- [ ] Discover post types: `curl https://albertmohler.com/wp-json/wp/v2/types`
- [ ] Add new secrets to GitHub repository

### Phase 1: WordPress Pipeline
- [ ] Update `config.py` with WordPress settings
- [ ] Create `scripts/09_extract_wp_content_v1.py`
- [ ] Create `scripts/10_chunk_wp_content_v1.py`
- [ ] Create `scripts/11_generate_wp_embeddings_v1.py`
- [ ] Create `scripts/12_upload_wp_to_pinecone_v1.py`
- [ ] Test full pipeline locally

### Phase 2: API Updates
- [ ] Modify `server/app.py` search endpoint for dual namespaces
- [ ] Add website result formatter function
- [ ] Update summarize endpoint for both sources
- [ ] Update stats endpoint to show both namespaces
- [ ] Test API responses

### Phase 3: Frontend
- [ ] Add section headers (YouTube / Website)
- [ ] Implement YouTube thumbnail grid (5 across)
- [ ] Implement website result cards
- [ ] Update JavaScript for new response structure
- [ ] Test responsive layout

### Phase 4: Automation
- [ ] Update `.github/workflows/daily-update.yml`
- [ ] Add GitHub secrets for WordPress
- [ ] Test manual workflow trigger
- [ ] Monitor first automated run

---

## Estimated Costs

### One-Time (Initial Indexing)

Assuming ~5,000 WordPress posts averaging 1,000 words each:
- **Total words**: 5,000,000
- **Chunks** (500 words each): ~10,000
- **Tokens** (1.3 tokens/word): ~6,500,000
- **Embedding cost**: ~$0.13 (at $0.02/1M tokens)

### Ongoing (Daily Updates)

Assuming ~5 new/updated posts per day:
- **Daily chunks**: ~10-20
- **Daily tokens**: ~10,000-20,000
- **Daily cost**: <$0.01

### Pinecone

- Same index, no additional cost
- Namespaces are free (just organizational)
- Current plan likely sufficient for combined load

---

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| WordPress API rate limiting | 0.5s delay between requests; use app password for higher limits |
| Large content corpus | Batch processing; robust progress tracking for resume |
| Inconsistent HTML formatting | BeautifulSoup with multiple fallback strategies |
| Podcast transcripts without timestamps | Detect format; use word-based chunking |
| Missing featured images | Fallback to category-based placeholder or site logo |
| Cache invalidation | Separate cache keys per namespace; consider shorter TTL initially |

---

## Future Enhancements

1. **Source filtering** - Let users filter results to YouTube-only or Website-only
2. **Category filtering** - Filter website results by category (Briefing, Culture, etc.)
3. **Date range** - Search within specific time periods
4. **Cross-reference** - Show related YouTube videos on article pages and vice versa
5. **Analytics** - Track which source users click on more
6. **Real-time sync** - WordPress webhook for instant indexing on publish

---

## File Reference

### Files to Create
```
scripts/09_extract_wp_content_v1.py
scripts/10_chunk_wp_content_v1.py
scripts/11_generate_wp_embeddings_v1.py
scripts/12_upload_wp_to_pinecone_v1.py
```

### Files to Modify
```
config.py
server/app.py
server/static/index.html
.github/workflows/daily-update.yml
```

### Data Directories to Create
```
data/wp_content/
data/wp_chunks/
data/wp_embeddings/
```

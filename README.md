# Albert Mohler YouTube Semantic Search

A semantic search pipeline for Albert Mohler's YouTube channel content. Extracts transcripts, generates embeddings, and enables AI-powered search via a Flask API.

## Overview

This project creates a searchable knowledge base from YouTube video transcripts:

1. **Extract** - Pull video IDs and metadata from the channel
2. **Transcribe** - Extract transcripts using rotating proxies
3. **Chunk** - Split transcripts into searchable segments (60-90 seconds each)
4. **Embed** - Generate vector embeddings via OpenAI
5. **Store** - Upload to Pinecone vector database
6. **Search** - Query via Flask API or WordPress plugin

## Prerequisites

- Python 3.8+
- API Keys:
  - [YouTube Data API v3](https://console.cloud.google.com/apis/credentials)
  - [OpenAI API](https://platform.openai.com/api-keys)
  - [Pinecone](https://app.pinecone.io/)
  - [Webshare](https://www.webshare.io/) (for transcript extraction proxies)

## Installation

```bash
# Clone the repository
git clone https://github.com/pastormiles/mohler.git
cd mohler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your API keys
```

## Configuration

All settings are in `config.py`. Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHANNEL_HANDLE` | @AlbertMohlerOfficial | YouTube channel handle |
| `TARGET_CHUNK_DURATION` | 75 seconds | Target chunk size |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `PINECONE_INDEX` | mohler-youtube | Pinecone index name |
| `SERVER_PORT` | 5007 | Flask API port |

## Pipeline Usage

Run scripts in order from the `scripts/` directory:

```bash
cd scripts/

# Stage 1: Extract video IDs from channel
python 01_extract_video_ids_v3.py

# Stage 2: Fetch detailed metadata
python 02_fetch_video_metadata_v3.py

# Stage 3: Extract transcripts (requires Webshare proxies)
python 03_extract_transcripts_v10.py

# Stage 4: Chunk transcripts
python 04_chunk_transcripts_v2.py

# Stage 5: Generate embeddings
python 05_generate_embeddings_v2.py

# Stage 6: Upload to Pinecone
python 06_upload_to_pinecone_v2.py

# Stage 7: Generate Flask API server
python 07_local_POC_v1.py

# Stage 8: Generate WordPress plugin
python 08_build_wp_plugin_v1.py
```

### Script Options

Most scripts support helpful flags:

```bash
# Limit processing (for testing)
python 03_extract_transcripts_v10.py --limit 10

# Retry failed items
python 03_extract_transcripts_v10.py --retry-blocked

# Incremental mode (only process new)
python 05_generate_embeddings_v2.py --incremental

# Test mode
python 03_extract_transcripts_v10.py --test
```

## Project Structure

```
mohler/
├── config.py              # Centralized configuration
├── .env                   # API keys (not in git)
├── .env.template          # Environment template
├── requirements.txt       # Python dependencies
├── scripts/               # Pipeline scripts (01-08)
├── data/
│   ├── video_ids/         # Extracted video IDs
│   ├── metadata/          # Video metadata
│   ├── transcripts/       # Raw transcripts
│   ├── chunks/            # Chunked transcripts
│   └── embeddings/        # Vector embeddings
├── server/                # Flask API (generated)
├── wp-plugin/             # WordPress plugin (generated)
└── logs/                  # Script logs
```

## Current Status

| Stage | Script | Status |
|-------|--------|--------|
| 1 | Extract Video IDs | Complete (722 videos) |
| 2 | Fetch Metadata | Complete |
| 3 | Extract Transcripts | Ready to run |
| 4 | Chunk Transcripts | Pending |
| 5 | Generate Embeddings | Pending |
| 6 | Upload to Pinecone | Pending |
| 7 | Generate Server | Ready |
| 8 | Generate WP Plugin | Ready |

## License

Private repository.

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

## Deployment (Google Cloud Run)

The API is designed for deployment on Google Cloud Run for automatic scaling and cost efficiency.

### Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
2. A GCP project with billing enabled
3. Cloud Run API enabled

### Initial Setup

```bash
# Authenticate with GCP
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Deploy

```bash
# Deploy to Cloud Run (from project root)
gcloud run deploy mohler-search \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=your-key,PINECONE_API_KEY=your-key"
```

Or use Secret Manager for API keys (recommended):

```bash
# Create secrets
echo -n "your-openai-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-pinecone-key" | gcloud secrets create pinecone-api-key --data-file=-

# Deploy with secrets
gcloud run deploy mohler-search \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest"
```

### Environment Variables

Set these in Cloud Run:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `API_KEY_POC` | No | API key for search UI |
| `API_KEY_WORDPRESS` | No | API key for WordPress plugin |
| `ADMIN_USERNAME` | No | Admin dashboard username |
| `ADMIN_PASSWORD` | No | Admin dashboard password |

### Estimated Costs

- **Under 2M requests/month**: Free tier covers it
- **2-3M requests/month**: ~$15-25/month

Cloud Run scales to zero when not in use, so you only pay for actual usage.

## Deployment (Render - Free Tier)

Render offers a free tier perfect for testing and low-traffic usage.

### Quick Deploy

1. Go to [render.com](https://render.com) and sign up/log in
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Render will auto-detect the Dockerfile
5. Configure settings:
   - **Name**: `mohler-search`
   - **Region**: Oregon (US West)
   - **Instance Type**: Free
6. Add environment variables:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `PINECONE_API_KEY` - Your Pinecone API key
7. Click **Deploy**

### Blueprint Deploy (Alternative)

Use the included `render.yaml` for one-click deployment:

1. Go to [render.com/deploy](https://render.com/deploy)
2. Enter your repository URL
3. Render will use `render.yaml` to configure the service
4. Add your API keys when prompted

### Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds (cold start)
- 750 hours/month of compute time
- Good for testing and demos

## Local Development

Run the server locally for testing:

```bash
# Generate server files first
cd scripts && python 07_local_POC_v1.py

# Start the server
cd ../server
python app.py

# Open http://localhost:5007
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Search interface |
| `/admin` | GET | Admin dashboard (requires auth) |
| `/api/v1/health` | GET | Health check |
| `/api/v1/search` | POST | Search transcripts |
| `/api/v1/summarize` | POST | AI summary of results |
| `/api/v1/stats` | GET | Index statistics |

## Automated Daily Updates (GitHub Actions)

The repository includes a GitHub Actions workflow that automatically checks for new videos and updates the search index daily at 2:00 AM Central Time.

### Setup

1. Go to your repository settings: `Settings → Secrets and variables → Actions`

2. Add these repository secrets:
   | Secret | Description |
   |--------|-------------|
   | `YOUTUBE_API_KEY` | YouTube Data API v3 key |
   | `OPENAI_API_KEY` | OpenAI API key for embeddings |
   | `PINECONE_API_KEY` | Pinecone API key |
   | `WEBSHARE_PROXY_USERNAME` | Webshare proxy username |
   | `WEBSHARE_PROXY_PASSWORD` | Webshare proxy password |

3. The workflow will run automatically. You can also trigger it manually from the Actions tab.

### What It Does

Each day at 2am:
1. Checks for new videos on the channel
2. Extracts transcripts for new videos
3. Chunks the transcripts
4. Generates embeddings via OpenAI
5. Uploads new vectors to Pinecone

Pipeline state is preserved between runs using GitHub Artifacts.

### Monitoring

- View workflow runs: `Actions` tab in GitHub
- Check logs for any failures
- Manual trigger available via "Run workflow" button

## License

Private repository.

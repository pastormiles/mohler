# Albert Mohler Semantic YouTube Search

## TODO (Priority Order)

1. **Convert Flask to FastAPI** - The current server (`server/app.py`) is Flask-based. Convert to FastAPI for:
   - Automatic Swagger UI at `/docs`
   - Async support for OpenAI/Pinecone API calls (better concurrent request handling)
   - Built-in Pydantic validation
   - Required for WordPress plugin deployment
   - Estimated time: ~1 hour

2. **WordPress plugin deployment** - After FastAPI conversion

3. **Build out CLAUDE.md** - Document full program architecture, data flow, file structure, API specs, deployment process, and all system components mapped out

---

## Project Overview

Semantic search across Albert Mohler's YouTube video transcripts using vector embeddings.

**Live URL**: https://am.nomion.ai

## Production Deployment

- **Platform**: Google Cloud Run
- **GCP Account**: miles@nomion.ai
- **Project**: "My First Project" (project ID: `project-115b3643-43f6-4ed0-816`)
- **Organization**: amohlerai-org
- **Service Name**: mohler-search
- **Region**: us-central1
- **Domain Mapping**: am.nomion.ai → mohler-search

**Deploy Command**:
```bash
gcloud run deploy mohler-search --source . --region us-central1 --project project-115b3643-43f6-4ed0-816
```

---

**Stack**:
- Flask API (to be converted to FastAPI)
- OpenAI embeddings (text-embedding-3-small)
- Pinecone vector database
- Google Cloud Run deployment

**Key Files**:
- `server/app.py` - Main API server
- `server/static/index.html` - Search UI
- `config.py` - Centralized configuration
- `scripts/01-08` - Data pipeline scripts

**Endpoints**:
- `GET /` - Search UI
- `POST /api/v1/search` - Search transcripts
- `POST /api/v1/summarize` - AI summary of results
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - Index statistics
- `GET /admin` - Admin dashboard (Basic Auth)

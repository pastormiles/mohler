# Render Deployment Guide

## Quick Deploy Steps

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click **New** → **Web Service**
3. Connect the GitHub repo: `pastormiles/mohler`
4. Render will auto-detect the Dockerfile
5. Configure:
   - **Name**: `mohler-search`
   - **Region**: Oregon (US West)
   - **Instance Type**: Free
6. Add environment variables:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `PINECONE_API_KEY` - Your Pinecone API key
7. Click **Deploy**

## Blueprint Deploy (Alternative)

The repo includes `render.yaml` for one-click deployment:

1. Go to [render.com/deploy](https://render.com/deploy)
2. Enter repository URL: `https://github.com/pastormiles/mohler`
3. Render will use `render.yaml` to configure the service
4. Add your API keys when prompted

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `API_KEY_POC` | No | Protect search UI with API key |
| `API_KEY_WORDPRESS` | No | API key for WordPress plugin |

## Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds (cold start)
- 750 hours/month of compute time
- Good for testing and demos

## Files Used

- `Dockerfile` - Docker configuration (same as Cloud Run)
- `render.yaml` - Render blueprint specification
- `server/app.py` - Flask application
- `config.py` - Configuration settings

## Post-Deploy

After deployment, your service will be available at:
```
https://mohler-search.onrender.com
```

Test the health endpoint:
```
curl https://mohler-search.onrender.com/api/v1/health
```

## Comparison: Render vs Cloud Run

| Feature | Render Free | Cloud Run |
|---------|-------------|-----------|
| Cold start | ~30 sec | ~5-10 sec |
| Sleep timeout | 15 min | Configurable |
| Free tier | 750 hrs/mo | 2M req/mo |
| Setup | Easier | More config |
| Best for | Testing | Production |

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router as api_router
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Exoplanet Candidate Analyst (ECA) API",
    description="Backend API serving the ECA machine learning predictions, knowledge graph traversal, literature RAG, and LangGraph agent.",
    version="1.0.0"
)

# Configure CORS for local development and production
# Set the FRONTEND_URL environment variable to your deployed Vercel frontend domain
# e.g. FRONTEND_URL=https://astro-candidate.vercel.app
frontend_origin = os.environ.get("FRONTEND_URL")
allow_origins = [frontend_origin] if frontend_origin else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Exoplanet Candidate Analyst (ECA) API",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)

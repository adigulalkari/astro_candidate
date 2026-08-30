# Exoplanet Candidate Analyst

An Agentic Machine Learning and Retrieval-Augmented Generation (RAG) system for the prioritization, analysis, and discovery of exoplanet candidates. 

This full-stack application provides an interactive LLM-powered research agent, multi-tool reasoning capabilities, tabular candidate ranking, and a Neo4j-backed knowledge graph to connect planetary data with scientific literature.

---

## 🏗 System Architecture

* **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons.
* **Backend:** FastAPI (Python), LangChain/LangGraph (Agent Orchestration).
* **Data Layer:** 
  * Tabular: SQLite (`planets.db`) / Parquet (`planets.parquet`)
  * Graph: Neo4j (Knowledge Graph for stars, planets, and papers)
  * Vector/RAG: Document retrieval over scientific literature (`papers.jsonl`)
* **ML Layer:** Custom scikit-learn/XGBoost pipelines for habitability and similarity scoring.

---

## 📁 Repository Structure

```
.
├── backend/
│   ├── app/
│   │   ├── agent/       # LangGraph execution graph and multi-step reasoning
│   │   ├── api/         # FastAPI routes and endpoint definitions
│   │   ├── ml/          # ML scoring and candidate ranking pipelines
│   │   ├── rag/         # Document retrieval and vector search capabilities
│   │   ├── services/    # External integrations (LLM providers, Neo4j driver)
│   │   └── tools/       # Agent tools (planet_search, ranking, literature, kg)
│   ├── data/            
│   │   ├── processed/   # SQLite and Parquet databases
│   │   └── raw/         # Source CSVs and JSONL literature datasets
│   ├── tests/           # Pytest integration and unit test suite
│   ├── main.py          # FastAPI application entry point
│   └── requirements.txt # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.tsx      # Main Agent Dashboard and chat UI
    │   ├── main.tsx     # React DOM entry
    │   └── index.css    # Tailwind entry point
    ├── package.json     # Node.js dependencies
    ├── vite.config.ts   # Vite bundler configuration
    └── tailwind.config.js
```

## 🛠 Local Development Setup
Prerequisites
Node.js (v18+)

Python (v3.9+)

Neo4j instance (Local Desktop or AuraDB)

LLM API Key (OpenAI, Anthropic, or local Ollama setup)

1. Backend Setup
Navigate to the backend directory and set up an isolated Python environment:
```
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


Environment Variables:
Create a .env file in the backend/ directory:

```
# LLM Provider
GEMINI_API_KEY=your_api_key_here

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Application Settings
CORS_ORIGINS=http://localhost:5173

```

Start the Development Server:
```
uvicorn main:app --reload --port 8000
```

The backend API will be available at `http://localhost:8000`.
Swagger UI documentation will be available at `http://localhost:8000/docs`

2. Frontend Setup
Open a new terminal window, navigate to the frontend directory, and install dependencies:
```
cd frontend
npm install
```
Environment Variables (Optional):
If you need to configure the API URL explicitly, create a .env in the frontend/ directory:
```
VITE_API_BASE_URL=http://localhost:8000
```

Start the Development Server:
```npm run dev```


### 📡 API Contract Overview
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/api/chat` | Main agentic interaction endpoint. Returns synthesized answers, executed reasoning steps, candidate scores, and supporting literature. |
| **GET** | `/api/planets` | Fetches tabular candidate data. Supports query params like `max_distance`, `max_radius`, and `discovery_method`. |
| **GET** | `/api/planets/{name}` | Returns detailed physical parameters for a specific star and planet. |
| **GET** | `/api/planets/{name}/graph` | Returns a serialized Neo4j node/edge subgraph for visualization. |
| **POST** | `/api/rank` | Accepts hypothetical planetary attributes (`pl_rade`, `pl_eqt`, etc.) and returns an ML habitability/priority score. |


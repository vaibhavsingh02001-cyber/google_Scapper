# Discovery Engine — AI-Powered Google Photos Retrieval Research Tool

A research pipeline that ingests public user feedback about Google Photos search/retrieval failures, classifies it using an LLM, clusters it into opportunity areas, and presents ranked insights via a hosted Streamlit dashboard.

## Quick Start

```bash
# 1. Clone & enter the project
cd discovery_engine

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up credentials
cp .env.example .env
# Edit .env with your API keys

# 5. Run data collection (Phase 1)
python run_collectors.py --sources all --limit 200

# 6. Run full pipeline
python run_pipeline.py

# 7. Launch dashboard
streamlit run app.py
```

## Project Structure

```
discovery_engine/
├── Docs/                  ← Design & planning documents
├── collectors/            ← One collector per data source
├── pipeline/              ← Normalizer, tagger, clusterer, scorer, reporter
├── models/                ← Pydantic schemas (RawItem, TaggedItem, OpportunityArea)
├── data/                  ← Raw & tagged datasets, Chroma vector store
├── reports/               ← Generated synthesis reports
├── logs/                  ← Collection and pipeline logs
├── app.py                 ← Streamlit dashboard
├── run_collectors.py      ← CLI for data collection only
├── run_pipeline.py        ← CLI for full pipeline (tag → cluster → report)
├── config.py              ← Centralised config loaded from .env
└── requirements.txt
```

## Pipeline Phases

| Phase | Command | Output |
|-------|---------|--------|
| Collect | `python run_collectors.py --sources all` | `data/raw_items.csv` |
| Tag | `python run_pipeline.py --step tag` | `data/tagged_items.json` |
| Cluster | `python run_pipeline.py --step cluster` | `data/opportunity_areas.csv` |
| Report | `python run_pipeline.py --step report` | `reports/synthesis_report.md` |
| All | `python run_pipeline.py` | All of the above |

## Required API Keys

| Key | Purpose | Source |
|-----|---------|--------|
| `ANTHROPIC_API_KEY` | LLM classification | console.anthropic.com |
| `OPENAI_API_KEY` | Text embeddings | platform.openai.com |
| `REDDIT_CLIENT_ID / SECRET` | Reddit data collection | reddit.com/prefs/apps |
| `YOUTUBE_API_KEY` | YouTube comments | console.cloud.google.com |

See `.env.example` for the complete list.

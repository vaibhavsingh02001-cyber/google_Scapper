# AI-Powered Discovery Engine for Google Photos Retrieval Research

> **Project Type:** Research Tool · Discovery Engine · Agentic Pipeline  
> **Owner:** Product Manager, Core Experience Team — Google Photos

---

## Role & Context

I am a Product Manager on the **Core Experience** team at Google Photos. Users accumulate thousands of photos, videos, screenshots, and documents over years of use. When they *remember* a photo exists but cannot describe it precisely (no date, place, or exact keywords), retrieval breaks down.

**Business Goal:** Increase the percentage of users who successfully retrieve a photo they remember but cannot precisely describe when they start searching.

> [!IMPORTANT]
> This is **NOT** a "make search better in general" project. The system required is a **research tool** that analyzes real user language about retrieval failure — to find the specific, evidence-backed opportunity area worth solving — **before** any product solution is designed.

---

## What to Build

An **AI-powered discovery engine**: a pipeline (agentic workflow, RAG system, or scripted LLM pipeline) that:

- Ingests public user feedback about photo retrieval
- Tags and clusters it along defined dimensions
- Produces a structured, queryable output plus a short synthesized report

The system must go beyond summarization and sentiment analysis — it must enable **comparison and ranking of distinct retrieval-problem clusters** using evidence volume and severity.

Build this as a **working, runnable system** (not a one-off analysis) with the following layers:

| # | Layer | Purpose |
|---|-------|---------|
| 1 | **Data Collection** | Ingest public user feedback from multiple platforms |
| 2 | **Tagging / Classification** | LLM-powered structured labeling per item |
| 3 | **Clustering / Aggregation** | Group tagged items into named opportunity areas |
| 4 | **Opportunity Comparison** | Rank clusters by evidence volume and severity |
| 5 | **Interface / Report** | Lightweight, shareable output for querying results |

---

## Data Sources (Public Only)

| Source | Notes |
|--------|-------|
| Google Play Store reviews | Google Photos app |
| Apple App Store reviews | Google Photos app |
| Reddit | r/googlephotos, r/android, r/photography, and relevant search threads |
| Google Photos Help Community | Support forum threads |
| YouTube comments | Google Photos review/tutorial videos |
| General web forums & social posts | "Can't find my photo" type complaints |

> [!NOTE]
> Respect each platform's terms of service and rate limits. Prefer official APIs or exports where available (e.g. Google Play review export, Reddit API/PRAW, YouTube Data API). **Do not bypass login walls or paywalls.**
>
> **Target corpus:** At least **500–1,000 relevant raw items** before filtering; aiming for a final filtered corpus in the **low hundreds to low thousands**.

---

## Analysis Requirements — Tagging Schema

For each piece of feedback, extract and tag using an **LLM classifier** (not keyword rules):

### 1. Relevance
Is this about **retrieval/search of existing photos** (not upload, editing, sharing, pricing, or unrelated bugs)?  
`yes` / `no`

### 2. Retrieval Problem Type
*Categorical — allow multi-label*

- Search returns irrelevant or no results
- Can't find screenshots, documents, or receipts specifically
- Can't narrow down or refine after a failed search
- Missing or wrong metadata (date, place, people, objects)
- People / face grouping or recognition problems
- Backup, sync, or storage issue (not true retrieval)
- Other / uncategorized

### 3. What the User Remembered About the Photo
*Free text + categorical*

`time/date` · `place` · `people present` · `event or occasion` · `object or visual detail` · `purpose/reason it was taken` · `emotion or mood` · `none stated`

### 4. What the User Explicitly Said They Forgot or Didn't Know

`exact date` · `exact location` · `filename/album` · `search keywords` · `nothing stated`

### 5. How They Tried to Search *(if described)*

- Exact keyword
- Vague description
- Browsing / scrolling manually
- Using filters (people / places / things)
- Asking someone else
- Gave up entirely

### 6. Failure Point in the Retrieval Journey
*Map to this funnel — tag which stage(s) the feedback evidences:*

```
(a) Could not express what they remembered in a query
(b) Search did not understand or match the clues given
(c) Results appeared but user could not judge/recognize the right one
(d) Search failed and user had no way to refine or narrow it
```

### 7. Severity / Sentiment

- Frustration level: `low` / `med` / `high`
- Did the user abandon the app or switch to a workaround? `yes` / `no`

### 8. Source Metadata

`platform` · `date (if available)` · `upvotes/likes (if available)` *(as a rough proxy for shared pain)*

---

## Analysis Requirements — Beyond Summarization

> [!IMPORTANT]
> Do **not** just summarize or run sentiment analysis. The pipeline must go deeper.

The pipeline must:

- **Cluster** tagged items into named **"opportunity areas"** — grouped by problem type + failure-point combination, not just keyword frequency.
- For each opportunity area, **compute**:
  - Evidence volume
  - Source diversity (how many platforms it appears on)
  - Estimated severity
- **Separate explicitly**:
  - VOICED complaints — users directly say "search doesn't work"
  - INFERRED patterns — e.g., many users describe purpose/mood-based memory but never explicitly ask for that kind of search *(this is the deeper insight the client wants)*
- Produce a **comparison table** ranking opportunity areas by business relevance (informed by volume + severity) — to distinguish best-evidenced vs. merely loudest.
- Surface **representative verbatim quotes** (short, properly attributed to source type, not fabricated) for each opportunity area.
- **Flag contradictions or open questions** the data cannot resolve — these become primary-research interview questions.

---

## Output Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| 1 | **Tagged Dataset** — every item with its labels, re-queryable | JSON or CSV |
| 2 | **Opportunity-Area Summary Table** — name, description, evidence volume, source breakdown, example quotes, business relevance | Markdown / CSV |
| 3 | **Synthesis Report** — top 3–5 opportunity areas, voiced vs. inferred gap, open questions for user research | Markdown / PDF (1–2 pages) |
| 4 | **Hosted / Shareable Interface** — a script, notebook, or minimal UI/dashboard that can be opened as a live link | Streamlit / Gradio / Web app |

---

## Suggested Tech Stack

| Layer | Tooling |
|-------|---------|
| **Collection** | Python scripts via platform APIs (Google Play Scraper, PRAW for Reddit, YouTube Data API) or n8n workflows for orchestration |
| **Classification / Tagging** | Claude (or another LLM) via API, using the tagging schema as a structured-output prompt (return JSON per item) |
| **Storage** | SQLite, CSV, or a lightweight vector DB (e.g., Chroma) if semantic search / RAG over the corpus is needed |
| **Aggregation / Clustering** | Python (pandas) for grouping and scoring; optionally embeddings + clustering for discovering unnamed patterns |
| **Interface** | Minimal Streamlit or Gradio app, or a static exported report — must be shareable as a link |

---

## Constraints

1. **Public data only** — no login-gated or scraped-behind-paywall content.
2. **No fabricated quotes or invented statistics** — every number in the output must trace back to the tagged dataset.
3. **Consistent classification schema** across all sources so cross-platform comparison is valid.
4. **This is a research/discovery tool, not the end product.** Do not design or build the actual retrieval feature yet — that comes after this analysis and primary user research.

---

## Definition of Done

- [ ] A **working, re-runnable pipeline** from raw scraped data → tagged dataset → opportunity-area comparison output.
- [ ] At least **one hosted / shareable artifact** that can be opened as a link.
- [ ] A **synthesis report** that names specific, evidence-backed opportunity areas (not "search should be better") with:
  - Volume, severity, and example quotes for each area
  - A clear statement of what is **voiced** vs. **inferred** from the data

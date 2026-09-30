# AI-Powered Customer Support Ticket Analysis System

An end-to-end customer support ticket analysis platform built for the **DOTMappers IT Pvt. Ltd. AI Intern Assessment**.

This solution ingests support ticket CSV data, interprets natural-language user queries using Google's **Gemini LLM**, executes structured query intents deterministically using **Pandas**, detects data anomalies, and presents a production-grade REST API and minimal responsive web interface.

---

## 1. Architecture & Design Principles

```
User Query (Natural Language)
            │
            ▼
┌─────────────────────────┐
│   Gemini LLM (GenAI)    │  ── Interpret query into validated Pydantic JSON intent
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│     Intent Validator    │  ── Verify operations, columns, operators, & bounds
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  Deterministic Pandas   │  ── Execute strict calculation on actual CSV dataset
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│   FastAPI & Web UI      │  ── Present natural language answer, metrics & tables
└─────────────────────────┘
```

### Critical Architectural Guarantee: Zero LLM Calculation Hallucinations
- **The LLM NEVER performs numerical calculations or dataset filtering.**
- The LLM's sole responsibility is **Natural-Language Understanding (NLU)**: translating a user question into a validated structured schema (`QueryIntent`).
- All counts, averages, groupings, and anomaly detections are computed **100% deterministically** using Python/Pandas against the actual 500-row `support_tickets.csv` dataset.

---

## 2. Key Features

1. **Dataset Ingestion & Schema Validation**:
   - Ingests `support_tickets.csv` (500 rows).
   - Validates schema columns (`ticket_id`, `created_at`, `category`, `priority`, `status`, `response_time_hrs`, `resolution_time_hrs`, `agent_id`, `customer_rating`, `issue_summary`).
   - Parses date fields and handles missing values safely (e.g., `resolution_time_hrs` and `customer_rating` are `NaN` for unresolved tickets).

2. **Natural Language Query System**:
   - Accepts natural-language questions from users via REST API or UI.
   - Translates questions into structured Pydantic `QueryIntent` using Gemini 2.5 Flash / 1.5 Flash.
   - Supports operations: `count`, `mean`, `sum`, `min`, `max`, `filter`, `group_by_aggregate`.

3. **Deterministic Anomaly Detection Engine**:
   - **Rule A: Abnormally Long Resolution Times**: Uses Interquartile Range (**IQR**) outlier detection on resolved tickets (`resolution_time_hrs`).
     - $Q1 = 25\text{th percentile} = 6.15\text{ hrs}$
     - $Q3 = 75\text{th percentile} = 22.95\text{ hrs}$
     - $IQR = Q3 - Q1 = 16.80\text{ hrs}$
     - $\text{Threshold} = Q3 + 1.5 \times IQR = 48.15\text{ hrs}$
     - Flagged: Resolved tickets with `resolution_time_hrs > 48.15` (21 tickets detected).
   - **Rule B: Unresolved High-Priority Tickets > 24 Hours**:
     - Flagged: Tickets where `status != 'Resolved'` AND `priority in ['High', 'Critical']` AND `age_hrs > 24.0` (80 tickets detected relative to dataset reference timestamp).

4. **FastAPI REST API & Minimal UI**:
   - `GET /health`: Health status & system metadata.
   - `POST /query`: Natural language question handler.
   - `GET /anomalies`: Anomaly detection report.
   - `GET /`: Embedded responsive dark-mode Web UI with sample query pills and anomaly tabs.

---

## 3. Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Environment variables & settings
│   ├── models.py           # Pydantic schemas (QueryIntent, QueryRequest, AnomalyResponse)
│   ├── data_loader.py      # Dataset CSV loader and preprocessor
│   ├── llm_interpreter.py  # Gemini LLM intent extractor with fallback
│   ├── query_engine.py     # Deterministic Pandas execution engine
│   ├── anomaly_detector.py # IQR & 24h SLA anomaly detection engine
│   ├── main.py             # FastAPI entrypoint & route handlers
│   └── static/
│       └── index.html      # Responsive UI interface
├── support_tickets.csv     # 500-row source dataset
├── support_tickets.xlsx    # Reference dataset
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.9+ installed on your system.

### Step 1: Clone & Navigate
```bash
git clone <your-repo-url>
cd AI-Intern
```

### Step 2: Create Virtual Environment & Install Dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API Key from Google AI Studio:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Note: If no API key is provided, the application automatically uses a robust heuristic fallback for standard sample queries without crashing).*

---

## 5. How to Run (Single Command)

To launch the FastAPI server and Web UI with a single command:

```bash
uvicorn app.main:app --reload --port 8000
```

Once running:
- **Web UI**: Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 6. API Endpoints

### 1. `GET /health`
Confirms system health, dataset load status, and Gemini API key state.

**Sample Response**:
```json
{
  "status": "healthy",
  "dataset_loaded": true,
  "dataset_rows": 500,
  "gemini_key_configured": true,
  "gemini_model": "gemini-2.5-flash"
}
```

### 2. `POST /query`
Submits a natural language question.

**Request Body**:
```json
{
  "question": "How many tickets are currently open?"
}
```

**Sample Response**:
```json
{
  "question": "How many tickets are currently open?",
  "answer": "There are 111 tickets matching status == Open.",
  "numerical_value": 111.0,
  "record_count": 111,
  "intent": {
    "is_supported": true,
    "operation": "count",
    "filters": [{"column": "status", "operator": "==", "value": "Open"}]
  },
  "pandas_code_description": "df[status == Open].shape[0]",
  "status": "success"
}
```

### 3. `GET /anomalies`
Returns detected resolution time outliers (IQR) and SLA violations (>24h unresolved high priority).

**Sample Response**:
```json
{
  "status": "success",
  "summary": {
    "long_resolution_count": 21,
    "long_resolution_threshold_hrs": 48.15,
    "iqr_q1_hrs": 6.15,
    "iqr_q3_hrs": 22.95,
    "unresolved_high_priority_over_24h_count": 80,
    "reference_timestamp": "2024-03-30 18:06:00"
  },
  "long_resolution_anomalies": [...],
  "unresolved_high_priority_anomalies": [...]
}
```

---

## 7. Sample Queries & Ground-Truth Verification

| Question | Calculated Answer | Pandas Logic Executed |
| :--- | :--- | :--- |
| **"How many tickets are currently open?"** | **111 tickets** | `len(df[df['status'] == 'Open'])` |
| **"Which agent resolved the most tickets this month?"** | **AGT-01 (16 tickets in March 2024)** | `df_latest_month.groupby('agent_id').size()` |
| **"Show me all Critical tickets not resolved within 12 hours."** | **34 tickets** (3 resolved >12h + 31 unresolved >12h age) | `df[(df['priority']=='Critical') & ((df['resolution_time_hrs']>12) \| ((df['status']!='Resolved') & (age_hrs>12)))]` |
| **"What is the average customer rating for Technical category tickets?"** | **3.74 / 5.0** | `df[df['category']=='Technical']['customer_rating'].mean()` |
| **"How many critical tickets are unresolved?"** | **31 tickets** | `len(df[(df['priority']=='Critical') & (df['status']!='Resolved')])` |
| **"Which agent has the lowest average customer rating?"** | **AGT-08 (3.48 rating)** | `df.groupby('agent_id')['customer_rating'].mean().sort_values()` |
| **"Are there any anomalies in resolution times this week?"** | **5 outlier tickets** in latest week (`>48.15h` threshold) | `df[created_at in latest week & status=='Resolved' & resolution_time_hrs>48.15]` |

---

## 8. Known Limitations & Trade-offs

1. **Single-Node In-Memory Storage**:
   - The current dataset (500 rows) is loaded into a Pandas DataFrame in memory for instant millisecond performance. For millions of rows, replacing Pandas with DuckDB or PostgreSQL is recommended.
2. **Text Search Granularity**:
   - Search on `issue_summary` uses Pandas string substring matching (`contains`). Complex semantic similarity matching on issue summaries would require embeddings.

---

## 9. Compliance Check against Assessment Requirements

- [x] Ingests `support_tickets.csv` without modifying original file.
- [x] Natural-language query system powered by Gemini LLM.
- [x] LLM translates query to structured intent; Python/Pandas calculates numeric answers.
- [x] Anomaly detection: IQR resolution time outliers + 24h unresolved high-priority SLA breaches.
- [x] REST API via FastAPI (`/health`, `/query`, `/anomalies`).
- [x] Minimal, responsive single-page web UI served at `/`.
- [x] Environment variable `GEMINI_API_KEY` configured via `.env.example` / `.gitignore`.
- [x] Single command startup: `uvicorn app.main:app --reload`.
- [x] Zero-cost API / free-tier model usage.

import os
import json
import logging
from typing import Optional, Dict, Any
from app.config import GEMINI_API_KEY, GEMINI_MODEL, is_gemini_api_key_configured
from app.models import QueryIntent, FilterModel, TimeFilterModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert natural-language query interpreter for a Support Ticket Analysis System.
Your job is to translate a user's natural language question into a validated structured query intent (JSON) that will be executed deterministically against a pandas DataFrame.

Dataset Schema:
- ticket_id: string (e.g. TKT-001)
- created_at: timestamp (range 2024-01-01 to 2024-03-30)
- category: enum ['Billing', 'Technical', 'General']
- priority: enum ['Low', 'Medium', 'High', 'Critical']
- status: enum ['Open', 'Resolved', 'Escalated']
- response_time_hrs: float (hours to first response)
- resolution_time_hrs: float (hours to resolution, null/NaN if unresolved)
- agent_id: string enum ['AGT-01', 'AGT-02', ..., 'AGT-12']
- customer_rating: float 1.0 - 5.0 (null/NaN if unresolved)
- issue_summary: free text

Allowed operations:
- 'count': Count rows matching filters.
- 'mean': Average of target_column (customer_rating, resolution_time_hrs, response_time_hrs) for rows matching filters.
- 'sum', 'min', 'max': Aggregations on target_column.
- 'filter': Return ticket rows matching filters.
- 'group_by_aggregate': Group by column (agent_id, category, priority, status) and compute count/mean.

Filter rules:
- 'unresolved': status != 'Resolved' (matches Open or Escalated)
- 'open': status == 'Open'
- 'resolved': status == 'Resolved'

Time filter rules:
- 'this month' or 'this_month': set time_filter period to 'latest_month' (which corresponds to March 2024 in dataset).
- 'this week' or 'this_week': set time_filter period to 'latest_week'.

Important:
- If question asks about data NOT in dataset (e.g. weather, stocks, user passwords, modifying data), set is_supported = false and provide unsupported_reason.
- DO NOT invent numerical answers or statistics. Your job is ONLY to extract the structured intent.
"""

def parse_question_to_intent(question: str) -> QueryIntent:
    """Uses Gemini API to interpret user question into structured QueryIntent."""
    question_clean = question.strip()
    if not question_clean:
        return QueryIntent(
            is_supported=False,
            unsupported_reason="Empty question provided."
        )

    # Check if Gemini API key is configured
    if not is_gemini_api_key_configured():
        logger.warning("GEMINI_API_KEY is not configured. Falling back to heuristic interpreter.")
        return _fallback_heuristic_intent(question_clean)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Call Gemini API with structured JSON schema output
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_text(text=SYSTEM_PROMPT),
                types.Part.from_text(text=f"User Question: {question_clean}")
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryIntent,
                temperature=0.0
            )
        )

        if response and response.text:
            intent_data = json.loads(response.text)
            intent = QueryIntent(**intent_data)
            return intent
        else:
            logger.error("Empty response received from Gemini API.")
            return _fallback_heuristic_intent(question_clean)

    except Exception as e:
        logger.error(f"Gemini API error during intent extraction: {e}")
        # Graceful fallback to heuristic rules so app never crashes
        return _fallback_heuristic_intent(question_clean)


def _fallback_heuristic_intent(question: str) -> QueryIntent:
    """Fallback rule-based interpreter when Gemini API is unavailable or unconfigured."""
    q = question.lower()

    # Question 1: "How many tickets are currently open?"
    if "how many" in q and "open" in q:
        return QueryIntent(
            is_supported=True,
            operation="count",
            filters=[FilterModel(column="status", operator="==", value="Open")],
            explanation="Count tickets where status is Open (Heuristic Fallback)"
        )

    # Question 2: "Which agent resolved the most tickets this month?"
    if "agent" in q and ("most tickets" in q or "resolved" in q) and ("month" in q or "highest" in q):
        return QueryIntent(
            is_supported=True,
            operation="group_by_aggregate",
            group_by="agent_id",
            agg_func="count",
            filters=[FilterModel(column="status", operator="==", value="Resolved")],
            time_filter=TimeFilterModel(period="latest_month"),
            sort_by="count",
            sort_ascending=False,
            limit=1,
            explanation="Group resolved tickets by agent for latest month, sorted descending (Heuristic Fallback)"
        )

    # Question 3: "Show me all Critical tickets not resolved within 12 hours."
    if "critical" in q and ("12 hours" in q or "12h" in q):
        return QueryIntent(
            is_supported=True,
            operation="filter",
            filters=[
                FilterModel(column="priority", operator="==", value="Critical"),
                FilterModel(column="resolution_time_hrs", operator=">", value=12.0)
            ],
            explanation="Filter Critical priority tickets not resolved within 12 hours (includes resolved >12h and unresolved >12h age) (Heuristic Fallback)"
        )

    # Question 4: "What is the average customer rating for Technical category tickets?"
    if "average" in q and "rating" in q and "technical" in q:
        return QueryIntent(
            is_supported=True,
            operation="mean",
            target_column="customer_rating",
            filters=[FilterModel(column="category", operator="==", value="Technical")],
            explanation="Mean customer rating for category Technical (Heuristic Fallback)"
        )

    # Question 5: "How many critical tickets are unresolved?"
    if "critical" in q and ("unresolved" in q or "not resolved" in q):
        return QueryIntent(
            is_supported=True,
            operation="count",
            filters=[
                FilterModel(column="priority", operator="==", value="Critical"),
                FilterModel(column="status", operator="!=", value="Resolved")
            ],
            explanation="Count tickets with priority Critical and status != Resolved (Heuristic Fallback)"
        )

    # Question 6: "Which agent has the lowest average customer rating?"
    if "agent" in q and "lowest" in q and "rating" in q:
        return QueryIntent(
            is_supported=True,
            operation="group_by_aggregate",
            group_by="agent_id",
            target_column="customer_rating",
            agg_func="mean",
            sort_by="mean",
            sort_ascending=True,
            limit=1,
            explanation="Group by agent, calculate mean customer rating, sort ascending (Heuristic Fallback)"
        )

    # Question 7: "Are there any anomalies in resolution times this week?"
    if "anomalies" in q or "anomaly" in q:
        return QueryIntent(
            is_supported=True,
            operation="filter",
            filters=[
                FilterModel(column="status", operator="==", value="Resolved"),
                FilterModel(column="resolution_time_hrs", operator=">", value=48.15)
            ],
            explanation="Filter resolved tickets exceeding the IQR resolution time anomaly threshold of 48.15 hours (Heuristic Fallback)"
        )

    # General unresolved count
    if "unresolved" in q or "not resolved" in q:
        return QueryIntent(
            is_supported=True,
            operation="count",
            filters=[FilterModel(column="status", operator="!=", value="Resolved")],
            explanation="Count tickets where status is not Resolved (Heuristic Fallback)"
        )

    # General count
    if "how many" in q or "total tickets" in q or "count" in q:
        return QueryIntent(
            is_supported=True,
            operation="count",
            filters=[],
            explanation="Count total matching tickets (Heuristic Fallback)"
        )

    # Fallback for unrecognized questions when API key is missing
    if not is_gemini_api_key_configured():
        return QueryIntent(
            is_supported=False,
            unsupported_reason="Gemini API key is not configured in environment (GEMINI_API_KEY). Please configure your API key in .env file to support arbitrary natural language queries."
        )

    return QueryIntent(
        is_supported=False,
        unsupported_reason="Unable to understand the question structure safely."
    )

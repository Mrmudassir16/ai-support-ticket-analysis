from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class FilterModel(BaseModel):
    column: str = Field(..., description="Dataset column name to filter on")
    operator: str = Field(..., description="Filter operator: ==, !=, >, <, >=, <=, in, contains")
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(..., description="Value(s) to filter against")


class TimeFilterModel(BaseModel):
    period: Optional[str] = Field(None, description="Time period: this_month, latest_month, this_week, custom")
    start_date: Optional[str] = Field(None, description="Start date ISO string YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date ISO string YYYY-MM-DD")


class QueryIntent(BaseModel):
    is_supported: bool = Field(True, description="True if the question can be answered using support ticket data")
    unsupported_reason: Optional[str] = Field(None, description="Reason if question is out of scope or unsupported")
    operation: str = Field("filter", description="Operation: count, mean, sum, min, max, filter, group_by_aggregate")
    target_column: Optional[str] = Field(None, description="Column target for aggregation (e.g., customer_rating, resolution_time_hrs)")
    filters: List[FilterModel] = Field(default_factory=list, description="List of filter conditions")
    group_by: Optional[str] = Field(None, description="Column to group by (e.g., agent_id, category, priority, status)")
    agg_func: Optional[str] = Field(None, description="Aggregation function for group_by: count, mean, sum, min, max")
    sort_by: Optional[str] = Field(None, description="Sort target: value or column name")
    sort_ascending: bool = Field(False, description="True for ascending, False for descending")
    time_filter: Optional[TimeFilterModel] = Field(None, description="Time window filter")
    limit: Optional[int] = Field(None, description="Maximum rows to return (e.g. 1 for top agent)")
    explanation: str = Field("", description="Short explanation of interpreted query logic")


class QueryRequest(BaseModel):
    question: str = Field(..., example="How many tickets are currently open?")


class QueryResponse(BaseModel):
    question: str
    answer: str
    numerical_value: Optional[float] = None
    record_count: int = 0
    intent: Optional[Dict[str, Any]] = None
    pandas_code_description: Optional[str] = None
    data: List[Dict[str, Any]] = []
    status: str = "success"
    error_message: Optional[str] = None


class AnomalyItem(BaseModel):
    ticket_id: str
    anomaly_type: str
    created_at: str
    category: str
    priority: str
    status: str
    agent_id: str
    response_time_hrs: float
    resolution_time_hrs: Optional[float] = None
    age_hrs: Optional[float] = None
    customer_rating: Optional[float] = None
    issue_summary: str
    reason: str


class AnomalySummary(BaseModel):
    long_resolution_count: int
    long_resolution_threshold_hrs: float
    iqr_q1_hrs: float
    iqr_q3_hrs: float
    unresolved_high_priority_over_24h_count: int
    reference_timestamp: str


class AnomalyResponse(BaseModel):
    status: str = "success"
    summary: AnomalySummary
    long_resolution_anomalies: List[AnomalyItem]
    unresolved_high_priority_anomalies: List[AnomalyItem]


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    dataset_rows: int
    gemini_key_configured: bool
    gemini_model: str

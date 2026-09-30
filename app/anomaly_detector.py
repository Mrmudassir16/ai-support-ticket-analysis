import pandas as pd
import numpy as np
from typing import List, Tuple
from app.data_loader import data_loader
from app.models import AnomalyResponse, AnomalySummary, AnomalyItem

def detect_anomalies() -> AnomalyResponse:
    """Detects anomalies in support tickets according to IQR outlier rules and age rules."""
    df = data_loader.get_df()

    # 1. Abnormally Long Resolution Times (IQR outlier detection)
    resolved_df = df[df["status"] == "Resolved"].dropna(subset=["resolution_time_hrs"]).copy()
    
    q1 = float(resolved_df["resolution_time_hrs"].quantile(0.25))
    q3 = float(resolved_df["resolution_time_hrs"].quantile(0.75))
    iqr = q3 - q1
    iqr_threshold = round(q3 + (1.5 * iqr), 2)

    long_res_df = resolved_df[resolved_df["resolution_time_hrs"] > iqr_threshold].sort_values(
        by="resolution_time_hrs", ascending=False
    )

    long_res_anomalies: List[AnomalyItem] = []
    for _, row in long_res_df.iterrows():
        res_time = round(float(row["resolution_time_hrs"]), 2)
        resp_time = round(float(row["response_time_hrs"]), 2)
        rating = round(float(row["customer_rating"]), 1) if pd.notna(row["customer_rating"]) else None
        
        long_res_anomalies.append(
            AnomalyItem(
                ticket_id=str(row["ticket_id"]),
                anomaly_type="abnormally_long_resolution",
                created_at=str(row["created_at"]),
                category=str(row["category"]),
                priority=str(row["priority"]),
                status=str(row["status"]),
                agent_id=str(row["agent_id"]),
                response_time_hrs=resp_time,
                resolution_time_hrs=res_time,
                customer_rating=rating,
                issue_summary=str(row["issue_summary"]),
                reason=f"Resolution time ({res_time} hrs) exceeds IQR upper bound threshold of {iqr_threshold} hrs (Q3={round(q3,2)}, IQR={round(iqr,2)})"
            )
        )

    # 2. Unresolved High-Priority Tickets Older than 24 Hours
    # We use max date in dataset as reference point so historical dataset logic evaluates accurately.
    max_dataset_time = df["created_at_dt"].max()
    unresolved_hc_df = df[
        (df["status"] != "Resolved") & 
        (df["priority"].isin(["High", "Critical"]))
    ].copy()

    unresolved_hc_df["age_hrs"] = (max_dataset_time - unresolved_hc_df["created_at_dt"]).dt.total_seconds() / 3600.0
    
    old_unresolved_df = unresolved_hc_df[unresolved_hc_df["age_hrs"] > 24.0].sort_values(
        by="age_hrs", ascending=False
    )

    unresolved_anomalies: List[AnomalyItem] = []
    for _, row in old_unresolved_df.iterrows():
        age_hrs = round(float(row["age_hrs"]), 1)
        resp_time = round(float(row["response_time_hrs"]), 2)
        
        unresolved_anomalies.append(
            AnomalyItem(
                ticket_id=str(row["ticket_id"]),
                anomaly_type="unresolved_high_priority_over_24h",
                created_at=str(row["created_at"]),
                category=str(row["category"]),
                priority=str(row["priority"]),
                status=str(row["status"]),
                agent_id=str(row["agent_id"]),
                response_time_hrs=resp_time,
                age_hrs=age_hrs,
                issue_summary=str(row["issue_summary"]),
                reason=f"Unresolved {row['priority']} ticket is {age_hrs} hours old (exceeds 24h SLA limit relative to dataset max timestamp {max_dataset_time.strftime('%Y-%m-%d %H:%M')})"
            )
        )

    summary = AnomalySummary(
        long_resolution_count=len(long_res_anomalies),
        long_resolution_threshold_hrs=iqr_threshold,
        iqr_q1_hrs=round(q1, 2),
        iqr_q3_hrs=round(q3, 2),
        unresolved_high_priority_over_24h_count=len(unresolved_anomalies),
        reference_timestamp=max_dataset_time.strftime("%Y-%m-%d %H:%M:%S")
    )

    return AnomalyResponse(
        status="success",
        summary=summary,
        long_resolution_anomalies=long_res_anomalies,
        unresolved_high_priority_anomalies=unresolved_anomalies
    )

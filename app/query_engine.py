import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.data_loader import data_loader
from app.models import QueryIntent, QueryResponse, FilterModel, TimeFilterModel

VALID_COLUMNS = {
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary", "created_at_dt"
}

def execute_query_intent(question: str, intent: QueryIntent) -> QueryResponse:
    """Executes a structured QueryIntent deterministically using Pandas."""
    if not intent.is_supported:
        return QueryResponse(
            question=question,
            answer=f"Question unsupported: {intent.unsupported_reason or 'Query out of scope for support ticket dataset.'}",
            intent=intent.model_dump(),
            status="unsupported",
            error_message=intent.unsupported_reason
        )

    try:
        df = data_loader.get_df()
        initial_count = len(df)
        filtered_df = df.copy()
        applied_conditions = []

        # 1. Apply Time Filter if specified
        if intent.time_filter:
            tf = intent.time_filter
            if tf.period == "latest_month" or tf.period == "this_month":
                max_month = filtered_df["created_at_dt"].dt.to_period("M").max()
                filtered_df = filtered_df[filtered_df["created_at_dt"].dt.to_period("M") == max_month]
                applied_conditions.append(f"created_at in latest month ({max_month})")
            elif tf.period == "latest_week" or tf.period == "this_week":
                max_week = filtered_df["created_at_dt"].dt.to_period("W").max()
                filtered_df = filtered_df[filtered_df["created_at_dt"].dt.to_period("W") == max_week]
                applied_conditions.append(f"created_at in latest week ({max_week})")
            elif tf.start_date:
                filtered_df = filtered_df[filtered_df["created_at_dt"] >= pd.to_datetime(tf.start_date)]
                applied_conditions.append(f"created_at >= {tf.start_date}")
            if tf.end_date:
                filtered_df = filtered_df[filtered_df["created_at_dt"] <= pd.to_datetime(tf.end_date)]
                applied_conditions.append(f"created_at <= {tf.end_date}")

        # 2. Apply Value Filters
        for f in intent.filters:
            col = f.column.strip().lower()
            # Map shorthand column names if LLM generated them
            col_map = {
                "resp_time_hrs": "response_time_hrs",
                "resol_time_hrs": "resolution_time_hrs",
                "cust_rating": "customer_rating",
                "rating": "customer_rating"
            }
            col = col_map.get(col, col)

            if col not in VALID_COLUMNS:
                continue

            op = f.operator.strip()
            val = f.value

            if op == "==":
                if isinstance(val, str):
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.lower() == val.lower()]
                else:
                    filtered_df = filtered_df[filtered_df[col] == val]
            elif op == "!=":
                if isinstance(val, str):
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.lower() != val.lower()]
                else:
                    filtered_df = filtered_df[filtered_df[col] != val]
            elif op == ">":
                val_num = float(val)
                if col == "resolution_time_hrs":
                    # Evaluates both resolved tickets taking > val_num hrs AND unresolved tickets whose age > val_num hrs
                    max_ref = df["created_at_dt"].max()
                    age_series = (max_ref - filtered_df["created_at_dt"]).dt.total_seconds() / 3600.0
                    is_res_over = (filtered_df["status"] == "Resolved") & (filtered_df["resolution_time_hrs"] > val_num)
                    is_unres_over = (filtered_df["status"] != "Resolved") & (age_series > val_num)
                    filtered_df = filtered_df[is_res_over | is_unres_over]
                else:
                    filtered_df = filtered_df[filtered_df[col] > val_num]
            elif op == "<":
                filtered_df = filtered_df[filtered_df[col] < float(val)]
            elif op == ">=":
                filtered_df = filtered_df[filtered_df[col] >= float(val)]
            elif op == "<=":
                filtered_df = filtered_df[filtered_df[col] <= float(val)]
            elif op == "in" and isinstance(val, list):
                val_clean = [str(v).lower() for v in val]
                filtered_df = filtered_df[filtered_df[col].astype(str).str.lower().isin(val_clean)]
            elif op == "contains" and isinstance(val, str):
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val, case=False, na=False)]

            applied_conditions.append(f"{col} {op} {val}")

        pandas_desc = f"df[{' & '.join(applied_conditions)}]" if applied_conditions else "df"
        op_type = intent.operation.lower()
        target_col = intent.target_column

        if target_col and target_col in ["resp_time_hrs", "resol_time_hrs", "cust_rating", "rating"]:
            target_map = {
                "resp_time_hrs": "response_time_hrs",
                "resol_time_hrs": "resolution_time_hrs",
                "cust_rating": "customer_rating",
                "rating": "customer_rating"
            }
            target_col = target_map[target_col]

        # 3. Perform Operations
        if op_type == "count":
            count_val = len(filtered_df)
            filter_text = f" matching {', '.join(applied_conditions)}" if applied_conditions else ""
            answer = f"There are {count_val} tickets{filter_text}."
            
            # Format output table
            data_records = _clean_records(filtered_df.head(50))
            return QueryResponse(
                question=question,
                answer=answer,
                numerical_value=float(count_val),
                record_count=count_val,
                intent=intent.model_dump(),
                pandas_code_description=f"{pandas_desc}.shape[0]",
                data=data_records,
                status="success"
            )

        elif op_type in ["mean", "average", "sum", "min", "max"]:
            col_to_agg = target_col if (target_col and target_col in VALID_COLUMNS) else "customer_rating"
            valid_series = filtered_df[col_to_agg].dropna()
            
            if len(valid_series) == 0:
                return QueryResponse(
                    question=question,
                    answer=f"No valid numerical values found for '{col_to_agg}' matching the filter criteria.",
                    numerical_value=None,
                    record_count=0,
                    intent=intent.model_dump(),
                    pandas_code_description=f"{pandas_desc}['{col_to_agg}'].{op_type}()",
                    data=[],
                    status="success"
                )

            if op_type in ["mean", "average"]:
                val = round(float(valid_series.mean()), 2)
                answer = f"The average {col_to_agg} is {val}."
            elif op_type == "sum":
                val = round(float(valid_series.sum()), 2)
                answer = f"The total sum of {col_to_agg} is {val}."
            elif op_type == "min":
                val = round(float(valid_series.min()), 2)
                answer = f"The minimum {col_to_agg} is {val}."
            elif op_type == "max":
                val = round(float(valid_series.max()), 2)
                answer = f"The maximum {col_to_agg} is {val}."

            data_records = _clean_records(filtered_df.head(50))
            return QueryResponse(
                question=question,
                answer=answer,
                numerical_value=val,
                record_count=len(filtered_df),
                intent=intent.model_dump(),
                pandas_code_description=f"{pandas_desc}['{col_to_agg}'].{op_type}()",
                data=data_records,
                status="success"
            )

        elif op_type == "group_by_aggregate":
            grp_col = intent.group_by if (intent.group_by and intent.group_by in VALID_COLUMNS) else "agent_id"
            agg_f = intent.agg_func or "count"
            
            if len(filtered_df) == 0:
                return QueryResponse(
                    question=question,
                    answer="No tickets match the specified criteria for grouping.",
                    record_count=0,
                    intent=intent.model_dump(),
                    data=[],
                    status="success"
                )

            if agg_f in ["mean", "average"] and target_col and target_col in VALID_COLUMNS:
                grouped = filtered_df.groupby(grp_col)[target_col].mean().round(2).reset_index()
                grouped.columns = [grp_col, f"mean_{target_col}"]
                sort_col = f"mean_{target_col}"
            else:
                grouped = filtered_df.groupby(grp_col).size().reset_index(name="count")
                sort_col = "count"

            # Sort
            asc = intent.sort_ascending
            grouped = grouped.sort_values(by=sort_col, ascending=asc)

            # Limit
            if intent.limit and intent.limit > 0:
                result_subset = grouped.head(intent.limit)
            else:
                result_subset = grouped

            top_row = result_subset.iloc[0]
            top_key = top_row[grp_col]
            top_val = top_row[sort_col]

            if sort_col == "count":
                answer = f"Agent {top_key} resolved the most tickets ({int(top_val)} tickets) matching criteria." if grp_col == "agent_id" else f"The group '{top_key}' has the highest ticket count ({int(top_val)})."
            else:
                answer = f"Agent {top_key} has the lowest average {target_col} of {top_val}." if (asc and grp_col == "agent_id") else f"Agent {top_key} has average {target_col} of {top_val}."

            data_records = result_subset.to_dict(orient="records")
            return QueryResponse(
                question=question,
                answer=answer,
                numerical_value=float(top_val),
                record_count=len(grouped),
                intent=intent.model_dump(),
                pandas_code_description=f"{pandas_desc}.groupby('{grp_col}').agg(...)",
                data=data_records,
                status="success"
            )

        else: # "filter" or general row retrieval
            limit = intent.limit or 50
            result_subset = filtered_df.head(limit)
            answer = f"Found {len(filtered_df)} tickets matching your search criteria. Showing top {len(result_subset)}."
            data_records = _clean_records(result_subset)

            return QueryResponse(
                question=question,
                answer=answer,
                record_count=len(filtered_df),
                intent=intent.model_dump(),
                pandas_code_description=pandas_desc,
                data=data_records,
                status="success"
            )

    except Exception as e:
        return QueryResponse(
            question=question,
            answer=f"An error occurred while executing the query: {str(e)}",
            intent=intent.model_dump() if intent else None,
            status="error",
            error_message=str(e)
        )


def _clean_records(df_slice: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts a pandas DataFrame slice into a clean JSON-serializable list of dicts."""
    records = []
    for _, row in df_slice.iterrows():
        rec = {}
        for col in df_slice.columns:
            if col == "created_at_dt":
                continue
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (np.integer, int)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = round(float(val), 2)
            else:
                rec[col] = str(val)
        records.append(rec)
    return records

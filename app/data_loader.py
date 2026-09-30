import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from app.config import CSV_FILE_PATH

class DataLoader:
    _instance: Optional['DataLoader'] = None
    _df: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance

    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Loads and prepares the support tickets CSV dataset."""
        target_path = csv_path or CSV_FILE_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Dataset CSV file not found at: {target_path}")

        df = pd.read_csv(target_path)
        
        # Expected column validation
        required_cols = [
            "ticket_id", "created_at", "category", "priority", 
            "status", "response_time_hrs", "resolution_time_hrs", 
            "agent_id", "customer_rating", "issue_summary"
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        # Parse timestamps correctly
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")

        # Ensure correct numeric data types
        df["response_time_hrs"] = pd.to_numeric(df["response_time_hrs"], errors="coerce")
        df["resolution_time_hrs"] = pd.to_numeric(df["resolution_time_hrs"], errors="coerce")
        df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

        # Clean string column whitespace
        string_cols = ["ticket_id", "category", "priority", "status", "agent_id", "issue_summary"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        self._df = df
        return df

    def get_df(self) -> pd.DataFrame:
        if self._df is None:
            return self.load_data()
        return self._df.copy()

data_loader = DataLoader()

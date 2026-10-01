import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.llm_interpreter import parse_question_to_intent
from app.query_engine import execute_query_intent
from app.models import QueryIntent


class TestTicketsPerCategoryQuery(unittest.TestCase):
    """Test suite validating the 'How many tickets are there per category?' query."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.expected_counts = {
            "General": 189,
            "Billing": 159,
            "Technical": 152
        }
        cls.total_expected_tickets = 500

    def test_api_query_tickets_per_category(self):
        """Tests POST /query endpoint with 'How many tickets are there per category?'"""
        response = self.client.post(
            "/query",
            json={"question": "How many tickets are there per category?"}
        )
        self.assertEqual(response.status_code, 200, "API should return HTTP 200 OK")
        
        data = response.json()
        self.assertEqual(data["status"], "success", "Response status should be 'success'")
        self.assertIsNotNone(data["data"], "Data payload should not be None")
        self.assertGreater(len(data["data"]), 0, "Data payload should contain records")

        # Convert returned data list into category count dict
        returned_counts = {row["category"]: row["count"] for row in data["data"]}
        
        # Validate ground truth category counts
        for category, expected_count in self.expected_counts.items():
            self.assertIn(category, returned_counts, f"Category '{category}' missing from results")
            self.assertEqual(
                returned_counts[category],
                expected_count,
                f"Count for '{category}' should be {expected_count}, got {returned_counts[category]}"
            )

        # Validate total sum equals dataset size (500)
        total_sum = sum(returned_counts.values())
        self.assertEqual(
            total_sum,
            self.total_expected_tickets,
            f"Sum of category counts ({total_sum}) must equal total dataset rows ({self.total_expected_tickets})"
        )

    def test_query_engine_deterministic_execution(self):
        """Tests direct deterministic pandas execution of group_by_aggregate on category."""
        intent = QueryIntent(
            is_supported=True,
            operation="group_by_aggregate",
            group_by="category",
            agg_func="count",
            sort_by="count",
            sort_ascending=False
        )
        res = execute_query_intent("How many tickets are there per category?", intent)
        
        self.assertEqual(res.status, "success")
        self.assertEqual(res.record_count, 3, "There should be exactly 3 categories")
        
        records = {r["category"]: r["count"] for r in res.data}
        self.assertEqual(records["General"], 189)
        self.assertEqual(records["Billing"], 159)
        self.assertEqual(records["Technical"], 152)

    def test_intent_parsing_for_category_query(self):
        """Tests that the question is correctly parsed into a category group_by intent."""
        question = "How many tickets are there per category?"
        intent = parse_question_to_intent(question)
        
        self.assertTrue(intent.is_supported, "Question should be supported")
        self.assertEqual(intent.operation, "group_by_aggregate")
        self.assertEqual(intent.group_by, "category")

    def test_synonymous_category_query_variations(self):
        """Tests variations of category breakdown questions."""
        variations = [
            "Tickets per category",
            "Show ticket counts by category",
            "What is the category breakdown of tickets?"
        ]
        for q in variations:
            with self.subTest(question=q):
                res = self.client.post("/query", json={"question": q})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                counts = {row["category"]: row["count"] for row in data["data"]}
                self.assertEqual(counts["General"], 189)


if __name__ == "__main__":
    unittest.main()

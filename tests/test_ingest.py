import unittest
from unittest.mock import Mock, patch

from complaint_intel.data.ingest import _extract_api_records, ingest_from_api


class ExtractApiRecordsTests(unittest.TestCase):
    def test_extracts_nested_hits(self) -> None:
        payload = {"hits": {"hits": [{"complaint_id": "1"}]}}
        self.assertEqual(_extract_api_records(payload), [{"complaint_id": "1"}])


class IngestFromApiTests(unittest.TestCase):
    @patch("complaint_intel.data.ingest.requests.Session")
    def test_stops_after_short_page(self, session_cls: Mock) -> None:
        session = session_cls.return_value
        response = Mock()
        response.json.return_value = {
            "hits": [
                {"complaint_id": "1", "product": "Credit card"},
                {"complaint_id": "2", "product": "Mortgage"},
            ]
        }
        response.raise_for_status.return_value = None
        session.get.return_value = response

        df = ingest_from_api("https://example.com/api", page_size=5, max_pages=3)

        self.assertEqual(len(df), 2)
        self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()

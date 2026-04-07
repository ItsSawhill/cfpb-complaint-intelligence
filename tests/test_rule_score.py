import unittest

import pandas as pd

from complaint_intel.risk.rule_score import compute_rule_risk


class ComputeRuleRiskTests(unittest.TestCase):
    def test_handles_missing_optional_structured_columns(self) -> None:
        df = pd.DataFrame(
            {
                "narrative": ["I want to dispute this fraud charge."],
                "date_received": ["2024-01-01"],
            }
        )

        out = compute_rule_risk(df)

        self.assertIn("risk_score_rule", out.columns)
        self.assertIn("risk_level_rule", out.columns)
        self.assertEqual(out.loc[0, "risk_level_rule"], "Low")
        self.assertGreater(out.loc[0, "n_keyword_hits"], 0)


if __name__ == "__main__":
    unittest.main()

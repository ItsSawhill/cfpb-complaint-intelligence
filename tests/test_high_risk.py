import unittest

import pandas as pd

from complaint_intel.risk.high_risk import define_high_risk


class DefineHighRiskTests(unittest.TestCase):
    def test_flags_disputed_untimely_and_vulnerable_complaints(self) -> None:
        df = pd.DataFrame(
            {
                "Consumer disputed?": ["Yes", "No", None],
                "Timely response?": ["Yes", "No", "Yes"],
                "Tags": [None, None, "Servicemember"],
            }
        )

        out = define_high_risk(df)

        self.assertEqual(out["high_risk"].tolist(), [1, 1, 1])
        self.assertIn("consumer disputed", out.loc[0, "high_risk_definition_reason"])
        self.assertIn("untimely response", out.loc[1, "high_risk_definition_reason"])
        self.assertIn("vulnerable consumer tag", out.loc[2, "high_risk_definition_reason"])


if __name__ == "__main__":
    unittest.main()

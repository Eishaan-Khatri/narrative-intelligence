import unittest

from system_a_discovery_engine.layer4_evaluation.ablation_runner import (
    generate_ablation_data,
    run_ablation,
)


class AblationMetricsTests(unittest.TestCase):
    def test_run_ablation_reports_real_recall_cutoffs(self):
        data = generate_ablation_data(n_users=24, n_items=80, seed=7)
        results = run_ablation(data=data, recall_ks=(10, 20, 50, 500), verbose=False)

        for column in ["Recall@10", "Recall@20", "Recall@50", "Recall@500"]:
            self.assertIn(column, results.columns)

        retrieval_rows = results[results["model"].isin(["M2", "M3", "M4"])]
        self.assertFalse(retrieval_rows["Recall@50"].isna().any())
        self.assertTrue((retrieval_rows["Recall@500"] >= retrieval_rows["Recall@50"]).all())


if __name__ == "__main__":
    unittest.main()

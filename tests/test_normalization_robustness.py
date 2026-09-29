"""Tests for within-corpus normalization robustness checks."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys
import unittest
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.normalization_robustness import compute_corpus_normalization_stats
from src.analysis.normalization_robustness import directionally_consistent
from src.analysis.normalization_robustness import run_normalization_robustness
from src.analysis.normalization_robustness import z_delta


class NormalizationRobustnessTests(unittest.TestCase):
    def test_corpus_normalization_stats_and_zero_variance(self) -> None:
        stats = compute_corpus_normalization_stats("demo", [0.2, 0.4, 0.6])
        self.assertAlmostEqual(stats.mean, 0.4)
        self.assertAlmostEqual(stats.std, 0.1632993161855452)
        self.assertAlmostEqual(z_delta(0.2, stats), 1.224744871391589)

        flat = compute_corpus_normalization_stats("flat", [0.5, 0.5])
        self.assertEqual(flat.std, 0.0)
        self.assertIsNone(z_delta(0.1, flat))

    def test_directional_consistency_rules(self) -> None:
        self.assertIs(directionally_consistent(0.1, "add_specific", 0.05), True)
        self.assertIs(directionally_consistent(-0.1, "add_specific", 0.05), False)
        self.assertIs(directionally_consistent(-0.1, "de_specify", 0.05), True)
        self.assertIs(directionally_consistent(0.04, "irrelevant_rewrite", 0.05), True)
        self.assertIs(directionally_consistent(0.06, "irrelevant_rewrite", 0.05), False)
        self.assertIsNone(directionally_consistent(0.0, "unknown", 0.05))

    def test_run_normalization_robustness_writes_summary_and_spearman_check(self) -> None:
        tmp_path = PROJECT_ROOT / f".tmp_normalization_robustness_{uuid.uuid4().hex}"
        tmp_path.mkdir()
        try:
            outputs = tmp_path / "outputs"
            (outputs / "speciteller").mkdir(parents=True)
            (outputs / "sentences").mkdir()
            (outputs / "features").mkdir()
            (outputs / "controlled_edits").mkdir()

            (outputs / "speciteller" / "demo_scores.tsv").write_text(
                "s1\t0.2\ns2\t0.4\ns3\t0.8\n",
                encoding="utf-8",
            )
            with (outputs / "sentences" / "demo.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"])
                writer.writerow(["demo", "a.md", "0", "Alpha one.", "s1"])
                writer.writerow(["demo", "a.md", "1", "Alpha two words.", "s2"])
                writer.writerow(["demo", "a.md", "2", "Alpha three words here.", "s3"])
            with (outputs / "features" / "demo_features.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "corpus_id",
                        "sent_id",
                        "tfidf_mean_nonzero",
                        "tfidf_max",
                        "technical_token_ratio",
                        "token_count",
                        "char_count",
                    ]
                )
                writer.writerow(["demo", "s1", "0.3", "0.6", "0.0", "2", "10"])
                writer.writerow(["demo", "s2", "0.2", "0.5", "0.1", "3", "16"])
                writer.writerow(["demo", "s3", "0.1", "0.4", "0.2", "4", "23"])
            with (outputs / "controlled_edits" / "demo_controlled_edit_scored.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "sent_id",
                        "corpus_id",
                        "sentence_original",
                        "speciteller_score_original",
                        "token_count",
                        "edit_type",
                        "sentence_edited",
                        "speciteller_score_edited",
                        "delta",
                    ]
                )
                writer.writerow(
                    ["s1", "demo", "Alpha one.", "0.2", "2", "add_specific", "Alpha one now.", "0.5", "0.3"]
                )
                writer.writerow(["s2", "demo", "Alpha two words.", "0.4", "3", "de_specify", "Alpha.", "0.1", "-0.3"])
                writer.writerow(
                    [
                        "s3",
                        "demo",
                        "Alpha three words here.",
                        "0.8",
                        "4",
                        "irrelevant_rewrite",
                        "Alpha here.",
                        "0.82",
                        "0.02",
                    ]
                )

            report = tmp_path / "analysis" / "normalization_robustness.md"
            out_csv = tmp_path / "analysis" / "normalization_robustness_controlled_edits.csv"
            result = run_normalization_robustness(
                corpus_ids=["demo"],
                outputs_root=outputs,
                report_path=report,
                csv_path=out_csv,
            )

            self.assertTrue(report.exists())
            self.assertTrue(out_csv.exists())
            self.assertEqual(len(result.controlled_edit_summaries), 3)
            self.assertTrue(all(check.invariant for check in result.spearman_checks))

            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            add_specific = next(row for row in rows if row["edit_type"] == "add_specific")
            self.assertEqual(add_specific["raw_directional_consistency"], "1.000000")
            self.assertGreater(float(add_specific["z_mean_delta"]), 0)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

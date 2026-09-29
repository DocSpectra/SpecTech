import csv
import json
from pathlib import Path


COMPACT = Path("analysis/round2_dgx_gptoss120b_human_first_analysis")


def _rows(name):
    with (COMPACT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows, **filters):
    matches = [row for row in rows if all(row[key] == value for key, value in filters.items())]
    assert len(matches) == 1
    return matches[0]


def test_compact_row_counts_and_manifest():
    expected = {
        "rating_order_distribution.csv": 54,
        "policy_summary.csv": 144,
        "policy_contrasts.csv": 126,
        "predictor_agreement.csv": 36,
        "proxy_confusion.csv": 18,
        "generator_comparison.csv": 66,
        "generator_policy_gain_contrasts.csv": 48,
    }
    for name, count in expected.items():
        assert len(_rows(name)) == count
    manifest = json.loads((COMPACT / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_rows"] == 180
    assert manifest["source_cases"] == 60
    assert manifest["score_counts"] == {"1": 36, "2": 31, "3": 43, "4": 34, "5": 35, "X": 1}
    assert manifest["bootstrap"]["replicates"] == 20000
    assert manifest["phase_e_authorized"] is False


def test_primary_policy_and_predictor_results():
    summaries = _rows("policy_summary.csv")
    unguided = _one(summaries, analysis_set="all_180", stratum_type="overall", policy="unguided_slot01")
    assert unguided["direction_valid_count"] == "31"
    spec = _one(summaries, analysis_set="all_180", stratum_type="overall", policy="speciteller_frozen_round1")
    assert spec["direction_valid_count"] == "34"
    consensus = _one(summaries, analysis_set="all_180", stratum_type="overall", policy="rank_consensus_secondary")
    assert consensus["direction_valid_count"] == "38"
    contrasts = _rows("policy_contrasts.csv")
    spec_gain = _one(contrasts, analysis_set="all_180", stratum_type="overall", policy="speciteller_frozen_round1")
    assert (spec_gain["valid_rate_difference"], spec_gain["difference_ci_low"], spec_gain["difference_ci_high"]) == ("0.05", "-0.06666666666666667", "0.16666666666666666")
    consensus_gain = _one(contrasts, analysis_set="all_180", stratum_type="overall", policy="rank_consensus_secondary")
    assert (consensus_gain["valid_rate_difference"], consensus_gain["difference_ci_low"], consensus_gain["difference_ci_high"]) == ("0.11666666666666667", "0.016666666666666666", "0.21666666666666667")
    predictors = _rows("predictor_agreement.csv")
    spec_rho = _one(predictors, stratum_type="overall", model="speciteller_frozen_round1")
    assert float(spec_rho["spearman_rho"]) > float(_one(predictors, stratum_type="overall", model="ko_run01")["spearman_rho"])
    assert {row["model"] for row in predictors if row["stratum_type"] == "overall"} >= {"ko_run01", "ko_run02", "ko_run03"}


def test_generator_and_proxy_results_are_complete_and_adverse_results_retained():
    generator = _rows("generator_comparison.csv")
    validity = _one(generator, comparison_family="per_source_candidate_summary", stratum_type="overall", estimand="valid_fraction", metric="")
    assert (validity["gpt_oss_120b_value"], validity["gemma4_12b_value"], validity["paired_difference_120b_minus_gemma"]) == ("0.511111111111", "0.761111111111", "-0.25")
    assert (validity["difference_ci_low"], validity["difference_ci_high"]) == ("-0.35", "-0.15")
    proxy = _one(_rows("proxy_confusion.csv"), analysis_set="all_180", stratum_type="overall")
    assert (proxy["true_accept"], proxy["false_accept"], proxy["false_reject"], proxy["true_reject"]) == ("43", "21", "49", "67")
    proxy_arm = _one(generator, comparison_family="automatic_proxy", stratum_type="overall", estimand="false_reject_rate", metric="")
    assert float(proxy_arm["paired_difference_120b_minus_gemma"]) > 0
    assert float(proxy_arm["difference_ci_low"]) > 0


def test_compact_privacy_and_bootstrap_counts():
    text = "\n".join(path.read_text(encoding="utf-8") for path in COMPACT.iterdir() if path.is_file()).casefold()
    for token in ("review_id", "candidate_id", "case_id", "source_sent_id", "sentence_original", "sentence_candidate"):
        assert token not in text
    for name in ("policy_summary.csv", "predictor_agreement.csv"):
        assert all(int(row["bootstrap_valid_replicates"]) == 20000 for row in _rows(name))
    generator = _rows("generator_comparison.csv")
    assert all(int(row["bootstrap_valid_replicates"]) > 0 for row in generator)

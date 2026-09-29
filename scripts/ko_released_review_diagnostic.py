"""Run and summarize the frozen released Yelp/Movie Ko diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ko_specificity.released_domain_diagnostic import (
    load_protocol,
    run_domain,
    summarize,
    verify_immutable_twitter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "run", "summarize"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ko_released_review_diagnostic_v1.json")
    )
    parser.add_argument("--domain", choices=("yelp", "movie"))
    parser.add_argument("--run-id", choices=("run01", "run02", "run03"))
    parser.add_argument("--attempt-id")
    parser.add_argument(
        "--run-root", type=Path, default=Path("outputs/round2/ko_released_review_diagnostic/runs")
    )
    parser.add_argument(
        "--compact-root", type=Path, default=Path("analysis/round2_ko_released_review_diagnostic")
    )
    args = parser.parse_args()
    repo_root = Path.cwd().resolve()
    protocol = load_protocol(args.config)
    if args.command == "verify":
        verified = verify_immutable_twitter(protocol, repo_root)
        print(f"verified {len(verified)} immutable Twitter files; config={protocol.sha256}")
        return
    if args.command == "run":
        if not args.domain or not args.run_id or not args.attempt_id:
            raise SystemExit("run requires --domain, --run-id, and --attempt-id")
        result = run_domain(
            protocol,
            domain=args.domain,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            repo_root=repo_root,
            run_root=args.run_root,
        )
        print(
            f"{args.domain}/{args.run_id}: {result['prediction_count']} predictions "
            f"{result['prediction_sha256']}"
        )
        return
    comparisons = summarize(
        protocol,
        repo_root=repo_root,
        run_root=args.run_root,
        compact_root=args.compact_root,
    )
    for row in comparisons:
        if row["observed_mean"] == "":
            print(f"{row['domain']} {row['metric']}: indeterminate (incomplete run set)")
        else:
            print(
                f"{row['domain']} {row['metric']}: {row['observed_mean']:.6f} "
                f"+/- {row['observed_population_std']:.6f} {row['classification']}"
            )


if __name__ == "__main__":
    main()

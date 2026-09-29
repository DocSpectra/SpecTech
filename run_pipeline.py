"""Entry point for the SpecTech specificity analysis pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import csv

from src.analysis.phase07_analysis_outputs import run_phase7_for_corpus
from src.analysis.score_stats import compute_and_write_score_stats_for_corpora
from src.ingestion.downloader import download_zip
from src.ingestion.downloader import ensure_git_repo
from src.ingestion.downloader import has_existing_download
from src.ingestion.loaders import load_documents
from src.ingestion.manifest import load_manifest
from src.ingestion.manifest import iter_document_paths
from src.features.proxies import compute_corpus_features
from src.features.proxies import write_feature_table
from src.normalization.router import normalize_document
from src.sentences.splitter import split_sentences
from src.sentences.table import build_sentence_records
from src.sentences.table import validate_sentence_records
from src.sentences.table import write_sentence_table
from src.speciteller.config import DEFAULT_SPECITELLER_CONFIG
from src.speciteller.runner import preflight_speciteller
from src.speciteller.runner import run_speciteller
from src.speciteller.tokenize import tokenize_for_speciteller
from src.utils.logging_utils import configure_logging
from src.utils.paths import ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpecTech specificity pipeline")
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help="Execution mode. Use pilot for small deterministic runs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest.yml"),
        help="Path to the corpus manifest YAML.",
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List discovered documents per corpus and exit.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=5,
        help="Max documents to list per corpus when using --list-docs.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download corpora sources for selected corpus IDs.",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all corpora listed in the manifest.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download corpora even if files already exist.",
    )
    parser.add_argument(
        "--corpus-ids",
        nargs="*",
        default=None,
        help="Optional list of corpus IDs to operate on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_directories()
    print(f"SpecTech pipeline initialized (mode={args.mode}, manifest={args.manifest}).")

    configs = load_manifest(args.manifest)
    if args.corpus_ids:
        configs = [cfg for cfg in configs if cfg.corpus_id in args.corpus_ids]

    if args.download or args.download_all:
        for config in configs:
            if not args.force_download and has_existing_download(config):
                print(f"\nSkipping {config.corpus_id} (already downloaded).")
                continue
            print(f"\nDownloading {config.corpus_id} ({config.source_type})")
            if config.source_type == "git":
                ensure_git_repo(config, force=args.force_download)
            elif config.source_type == "html_zip":
                zip_path = download_zip(config, dest_name="corpus.zip", force=args.force_download)
                print(f"Downloaded zip to {zip_path}")
            elif config.source_type == "wiki_dump":
                print("Wikipedia dump download not automated; place files in data/corpora.")
            else:
                print(f"Unknown source_type: {config.source_type}")

        print("\nDownload validation:")
        for config in configs:
            has_docs = any(iter_document_paths(config))
            status = "OK" if has_docs else "MISSING"
            print(f"- {config.corpus_id}: {status}")

    if args.list_docs:
        for config in configs:
            print(f"\n[{config.corpus_id}] -> {config.local_path}")
            count = 0
            for path in iter_document_paths(config):
                print(f"  - {path}")
                count += 1
                if count >= args.list_limit:
                    break
        return

    if args.mode in {"pilot", "full"}:
        preflight_ok = False
        for config in configs:
            records = []
            for doc in load_documents(config):
                normalized = normalize_document(doc.doc_path, doc.content)
                sentences = split_sentences(normalized)
                records.extend(
                    build_sentence_records(config.corpus_id, str(doc.doc_path), sentences)
                )
            validate_sentence_records(records)
            output_path = Path("outputs") / "sentences" / f"{config.corpus_id}.csv"
            write_sentence_table(output_path, records)
            spec_input = Path("outputs") / "speciteller" / f"{config.corpus_id}.tsv"
            spec_output = Path("outputs") / "speciteller" / f"{config.corpus_id}_scores.tsv"
            spec_input.parent.mkdir(parents=True, exist_ok=True)
            with spec_input.open("w", encoding="utf-8") as handle:
                for record in records:
                    tokenized = tokenize_for_speciteller(record.sent_text)
                    handle.write(f"{record.sent_id}\t{tokenized}\n")

            # Cache behavior: if output exists and line counts match input, skip rescoring.
            expected = len(records)
            should_run = True
            if spec_output.exists():
                existing = sum(1 for _ in spec_output.open("r", encoding="utf-8") if _.strip())
                if existing == expected:
                    should_run = False

            if should_run:
                if not preflight_ok:
                    preflight_speciteller(DEFAULT_SPECITELLER_CONFIG)
                    preflight_ok = True
                run_speciteller(DEFAULT_SPECITELLER_CONFIG, spec_input, spec_output)

            # Join integrity validation
            sent_ids = {record.sent_id for record in records}
            score_ids: set[str] = set()
            with spec_output.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter="\t")
                for row in reader:
                    if not row:
                        continue
                    if len(row) != 2:
                        raise ValueError(f"Malformed SpeciTeller score row: {row}")
                    score_ids.add(row[0])
            if sent_ids != score_ids:
                missing = len(sent_ids - score_ids)
                extra = len(score_ids - sent_ids)
                raise ValueError(
                    f"SpeciTeller join mismatch for {config.corpus_id}: missing={missing}, extra={extra}"
                )

            feature_rows = compute_corpus_features(records)
            feature_output = Path("outputs") / "features" / f"{config.corpus_id}_features.csv"
            write_feature_table(feature_output, feature_rows)
            run_phase7_for_corpus(config.corpus_id, outputs_root=Path("outputs"))

        compute_and_write_score_stats_for_corpora(
            corpus_ids=[cfg.corpus_id for cfg in configs],
            outputs_root=Path("outputs"),
        )

        print(
            "Sentence tables + SpeciTeller scores + feature tables + analysis outputs + "
            "score summary stats written to outputs/."
        )
        return

    print("Phase 1+ pipeline steps will be added in subsequent phases.")


if __name__ == "__main__":
    main()
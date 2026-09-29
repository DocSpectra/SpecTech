"""Compute the frozen released-domain metrics (Python 3.6 compatible)."""
from __future__ import print_function

import argparse
import json
import math

import numpy
from scipy.stats import kendalltau, spearmanr


def parse_prediction(line):
    value = line.strip()
    if value.startswith("tensor(") and value.endswith(")"):
        value = value[7:-1]
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError("invalid prediction: {}".format(value))
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--domain", choices=("yelp", "movie"), required=True)
parser.add_argument("--predictions", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--expected-count", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.predictions) as handle:
    predicted = numpy.asarray([parse_prediction(line) for line in handle if line.strip()])
with open(args.labels) as handle:
    labels = numpy.asarray([(float(line.strip()) - 1.0) / 4.0 for line in handle])

labels = labels[1:]
if len(predicted) != args.expected_count or len(labels) != args.expected_count:
    raise ValueError(
        "prediction/label mismatch: predictions={} labels={} expected={}".format(
            len(predicted), len(labels), args.expected_count
        )
    )

result = {
    "schema_version": "ko_released_review_run_metrics_v1",
    "domain": args.domain,
    "prediction_count": int(len(predicted)),
    "withheld_annotated_rows": 1,
    "label_transform": "(released_label - 1) / 4",
    "spearman": float(spearmanr(labels, predicted)[0]),
    "kendall_tau": float(kendalltau(labels, predicted)[0]),
    "mae": float(numpy.mean(numpy.abs(labels - predicted))),
    "prediction_mean": float(numpy.mean(predicted)),
    "prediction_population_std": float(numpy.std(predicted)),
    "prediction_min": float(numpy.min(predicted)),
    "prediction_max": float(numpy.max(predicted)),
}
with open(args.output, "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")

"""Validate released-domain predictions against the paper's Twitter metrics."""
from __future__ import print_function

import argparse
import json

import numpy
from scipy.stats import kendalltau, spearmanr


def parse_prediction(line):
    value = line.strip()
    if value.startswith("tensor(") and value.endswith(")"):
        value = value[7:-1]
    return float(value)


parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
parser.add_argument("--labels", default="/opt/ko/dataset/data/twitterv.txt")
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.predictions) as handle:
    predicted = numpy.asarray([parse_prediction(line) for line in handle if line.strip()])
with open(args.labels) as handle:
    labels = numpy.asarray([(float(line.strip()) - 1.0) / 4.0 for line in handle])
# The official tv=1 loader withholds the first released test row.
labels = labels[1:]
if len(predicted) != len(labels):
    raise ValueError("prediction/label mismatch: {} != {}".format(len(predicted), len(labels)))
result = {
    "schema_version": "ko_released_twitter_validation_v1",
    "rows": int(len(predicted)),
    "spearman": float(spearmanr(labels, predicted)[0]),
    "kendall_tau": float(kendalltau(labels, predicted)[0]),
    "mae": float(numpy.mean(numpy.abs(labels - predicted))),
    "paper_three_run_mean": {"spearman": 0.676, "kendall_tau": 0.487, "mae": 0.113},
    "paper_three_run_sd": {"spearman": 0.004, "kendall_tau": 0.005, "mae": 0.001},
}
with open(args.output, "w") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")

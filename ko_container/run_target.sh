#!/usr/bin/env bash
set -euo pipefail

mode="${1:-released-smoke}"
glove=/artifacts/glove.840B.300d.txt
expected_glove_sha="${GLOVE_TXT_SHA256:-9641ef45628fd8c5c0bc555c8944f7bad7bce36ae7be3cafff0bb109fe60883e}"

if [[ ! -f "$glove" ]]; then
  echo "missing external GloVe file: $glove" >&2
  exit 20
fi
actual_glove_sha="$(sha256sum "$glove" | cut -d' ' -f1)"
if [[ "$actual_glove_sha" != "$expected_glove_sha" ]]; then
  echo "GloVe checksum mismatch: expected=$expected_glove_sha actual=$actual_glove_sha" >&2
  exit 21
fi

cd /work
rm -rf /work/ko
cp -a /opt/ko /work/ko
cd /work/ko
ln -s "$glove" glove.840B.300d.txt
mkdir -p savedir /output

case "$mode" in
  target)
    for name in twitters.txt twitteru.txt twitterl.txt twitterv.txt row_map.csv; do
      if [[ ! -f "/target/$name" ]]; then
        echo "missing target bundle file: $name" >&2
        exit 22
      fi
    done
    cp /target/twitters.txt /target/twitteru.txt /target/twitterl.txt /target/twitterv.txt dataset/data/
    train_args=()
    ;;
  released-smoke)
    train_args=(--esize 32 --uss 32 --sss 1 --me 2)
    ;;
  released-full)
    train_args=()
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 23
    ;;
esac

python train.py --gpu_id 0 --test_data twitter "${train_args[@]}" 2>&1 | tee /output/train.log
python test.py --gpu_id 0 --test_data twitter 2>&1 | tee /output/test.log
cp predictions.txt /output/predictions.txt
sha256sum savedir/3osmodel.pickle > /output/model.sha256
if [[ "$mode" == "target" ]]; then
  cp savedir/3osmodel.pickle /output/model.pickle
fi
python /opt/spectech/write_run_metadata.py \
  --mode "$mode" \
  --predictions /output/predictions.txt \
  --model savedir/3osmodel.pickle \
  --output /output/run_metadata.json

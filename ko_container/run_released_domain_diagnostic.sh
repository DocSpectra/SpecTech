#!/usr/bin/env bash
set -euo pipefail

domain="${1:?domain is required}"
glove=/artifacts/glove.840B.300d.txt
expected_glove_sha=9641ef45628fd8c5c0bc555c8944f7bad7bce36ae7be3cafff0bb109fe60883e

case "$domain" in
  yelp)
    sentence_file=yelps.txt
    label_file=yelpv.txt
    binary_file=yelpl.txt
    unlabeled_file=yelpu.txt
    expected_sentence_sha=fdb519e84b85065416098e499a709b780f33a4a0b9ddaa293a6531fc269666b3
    expected_label_sha=8e2a958a8f86c881046a5ec02c78a15e946d578bfea721555d978347cc00379f
    expected_binary_sha=a5c195158d4a7e1bbfe7351a086bfc4bc1911167e4fbb508df362fe51ca46872
    expected_unlabeled_sha=025a89b41476cc9be102384b079b68700c61f663e8e67aee3163e322c3447cc0
    expected_annotated_rows=845
    expected_unlabeled_rows=95650
    expected_prediction_rows=844
    ;;
  movie)
    sentence_file=movies.txt
    label_file=moviev.txt
    binary_file=moviel.txt
    unlabeled_file=movieu.txt
    expected_sentence_sha=32502bfac5169c2c71bc77ae9451ae04ad62c1725cdd2f6939406ab9152f8811
    expected_label_sha=1265def85fafe1429a4d36b6f51ce36c99107c2def3db490062ea3e8d91e55ba
    expected_binary_sha=3418b8c0e0bc52f67d2bb1b3509fdf89a532f869cf86a3f0c83a772d275a32fe
    expected_unlabeled_sha=b01025db8961a4da9ad05b720392e93ead9fed96e9db70314374a0e3e74b9fa7
    expected_annotated_rows=920
    expected_unlabeled_rows=11855
    expected_prediction_rows=919
    ;;
  *)
    echo "unsupported frozen domain: $domain" >&2
    exit 30
    ;;
esac

if [[ ! -f "$glove" ]]; then
  echo "missing external GloVe file" >&2
  exit 31
fi
echo "$expected_glove_sha  $glove" | sha256sum -c -

cd /opt/ko
echo "$expected_sentence_sha  dataset/data/$sentence_file" | sha256sum -c -
echo "$expected_label_sha  dataset/data/$label_file" | sha256sum -c -
echo "$expected_binary_sha  dataset/data/$binary_file" | sha256sum -c -
echo "$expected_unlabeled_sha  dataset/data/$unlabeled_file" | sha256sum -c -
[[ "$(wc -l < "dataset/data/$sentence_file")" -eq "$expected_annotated_rows" ]]
[[ "$(wc -l < "dataset/data/$label_file")" -eq "$expected_annotated_rows" ]]
[[ "$(wc -l < "dataset/data/$binary_file")" -eq "$expected_annotated_rows" ]]
[[ "$(wc -l < "dataset/data/$unlabeled_file")" -eq "$expected_unlabeled_rows" ]]

cd /work
rm -rf /work/ko
cp -a /opt/ko /work/ko
cd /work/ko
ln -s "$glove" glove.840B.300d.txt
mkdir -p savedir /output

python train.py --gpu_id 0 --test_data "$domain" 2>&1 | tee /output/train.log
python test.py --gpu_id 0 --test_data "$domain" 2>&1 | tee /output/test.log
cp predictions.txt /output/predictions.txt
cp savedir/3osmodel.pickle /output/teacher_model.pickle
sha256sum /output/teacher_model.pickle > /output/model.sha256
python /diagnostic/validate_released_domain.py \
  --domain "$domain" \
  --predictions /output/predictions.txt \
  --labels "dataset/data/$label_file" \
  --expected-count "$expected_prediction_rows" \
  --output /output/metrics.json
python /opt/spectech/write_run_metadata.py \
  --mode "released-$domain-full" \
  --predictions /output/predictions.txt \
  --model /output/teacher_model.pickle \
  --output /output/container_run_metadata.json

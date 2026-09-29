param(
    [string]$Volume = "spectech-ko-glove-840b-v1"
)
$ErrorActionPreference = "Stop"
docker volume create $Volume | Out-Null
docker run --rm --volume "${Volume}:/artifacts" `
  python:3.6.15-buster@sha256:08c787fde0ef9fac8d9244e0d49939ce3dfca27198fac37ac2748c2d3c8fc987 `
  bash -lc "set -euo pipefail; cd /artifacts; if [ ! -f glove.840B.300d.zip ]; then curl -fL --retry 3 -o glove.840B.300d.zip https://nlp.stanford.edu/data/glove.840B.300d.zip; fi; echo 'c06db255e65095393609f19a4cfca20bf3a71e20cc53e892aafa490347e3849f  glove.840B.300d.zip' | sha256sum -c -; if [ ! -f glove.840B.300d.txt ]; then unzip -q glove.840B.300d.zip; fi; echo '9641ef45628fd8c5c0bc555c8944f7bad7bce36ae7be3cafff0bb109fe60883e  glove.840B.300d.txt' | sha256sum -c -"

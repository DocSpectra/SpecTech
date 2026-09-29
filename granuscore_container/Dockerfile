FROM pytorch/pytorch@sha256:60f22fb80755fd0b470fb47928dbd55816aa9f847edd95cf43c93253507a9ddf

ARG GRANUSCORE_WHEEL_URL=https://files.pythonhosted.org/packages/f2/a6/777dcba572530f1f2c2442954eccecefc093a49a053406f4d3e17dda70a9/granuscore-1.0.1-py3-none-any.whl
ARG GRANUSCORE_WHEEL_SHA256=8d0325512243ef5e1fb962ccbdd65e61110f5fd165de699274932cb18d747d0c
ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
ARG SPACY_MODEL_SHA256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
ARG GRANUSCORE_MODEL=Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun
ARG GRANUSCORE_MODEL_REVISION=b170cbfa5bb770f144c69f75f826ccbd6e0c7b53

LABEL org.opencontainers.image.title="SpecTech GranuScore 1.0.1 experiment" \
      org.opencontainers.image.source="https://github.com/lukasellinger/granuscore" \
      org.opencontainers.image.revision="73438cdbb06ab72ec2a50507a0db62c352ec8169" \
      org.opencontainers.image.licenses="MIT" \
      spectech.granuscore.model.revision="b170cbfa5bb770f144c69f75f826ccbd6e0c7b53"

ENV XDG_CACHE_HOME=/opt/granuscore/cache \
    GRANUSCORE_MODEL_PATH=/opt/granuscore/model \
    GRANUSCORE_IMAGE_ID=spectech-granuscore:1.0.1-hit-wordnet-v1

COPY granuscore_container/constraints.txt /opt/granuscore/constraints.txt
COPY granuscore_container/fetch_verified.py /opt/granuscore/fetch_verified.py
COPY granuscore_container/verify_published_examples.py /opt/granuscore/verify_published_examples.py
COPY granuscore_container/score_csv.py /opt/granuscore/score_csv.py

RUN set -eux; \
    python /opt/granuscore/fetch_verified.py "${GRANUSCORE_WHEEL_URL}" /tmp/granuscore-1.0.1-py3-none-any.whl "${GRANUSCORE_WHEEL_SHA256}"; \
    python /opt/granuscore/fetch_verified.py "${SPACY_MODEL_URL}" /tmp/en_core_web_sm-3.8.0-py3-none-any.whl "${SPACY_MODEL_SHA256}"; \
    python -m pip install --no-cache-dir --constraint /opt/granuscore/constraints.txt \
      /tmp/granuscore-1.0.1-py3-none-any.whl /tmp/en_core_web_sm-3.8.0-py3-none-any.whl; \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('${GRANUSCORE_MODEL}', revision='${GRANUSCORE_MODEL_REVISION}', local_dir='/opt/granuscore/model')"; \
    python /opt/granuscore/verify_published_examples.py --tolerance 0.05; \
    python -m pip check; \
    rm /tmp/granuscore-1.0.1-py3-none-any.whl /tmp/en_core_web_sm-3.8.0-py3-none-any.whl

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /work
ENTRYPOINT ["python", "/opt/granuscore/score_csv.py"]

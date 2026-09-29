FROM python:3.6.15-buster@sha256:08c787fde0ef9fac8d9244e0d49939ce3dfca27198fac37ac2748c2d3c8fc987

ARG KO_COMMIT=36f8e835e9dc6087d5b6763accf302db175947b1
ARG KO_ARCHIVE_SHA256=4c1e9bead56684f8f74f2fef0462bdfb7ccbc4de7cb714b0e4ad1575be567892
ARG KO_NO_CUDA_SHA256=623fd4d762d5eaf45c034a4c10759babeebd374575d3d1ff53fcfe050d3604f5
ARG TORCH_WHEEL_SHA256=0373ca29af27a0232536f73b233e1298b036c8390b9abf559641bc79644b1761
ARG NUMPY_WHEEL_SHA256=27e11c7a8ec9d5838bc59f809bfa86efc8a4fd02e58960fa9c49d998e14332d5
ARG SCIPY_WHEEL_SHA256=f31338ee269d201abe76083a990905473987371ff6f3fdb76a3f9073a361cf37

LABEL org.opencontainers.image.title="Local Ko et al. 2019 reproduction environment" \
      org.opencontainers.image.revision="36f8e835e9dc6087d5b6763accf302db175947b1" \
      org.opencontainers.image.licenses="NOASSERTION" \
      spectech.redistribution="prohibited-unless-upstream-permission-is-obtained"

RUN set -eux; \
    curl -fL --retry 3 -o /tmp/ko.tar.gz \
      "https://github.com/wjko2/Domain-Agnostic-Sentence-Specificity-Prediction/archive/${KO_COMMIT}.tar.gz"; \
    echo "${KO_ARCHIVE_SHA256}  /tmp/ko.tar.gz" | sha256sum -c -; \
    mkdir -p /opt/ko; \
    tar -xzf /tmp/ko.tar.gz --strip-components=1 -C /opt/ko; \
    echo "${KO_NO_CUDA_SHA256}  /opt/ko/no_cuda.zip" | sha256sum -c -; \
    unzip -o /opt/ko/no_cuda.zip -d /opt/ko; \
    rm /tmp/ko.tar.gz

RUN set -eux; \
    curl -fL --retry 3 -o /tmp/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
      https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl; \
    mkdir /tmp/wheels; \
    python -m pip download --no-deps --only-binary=:all: --dest /tmp/wheels numpy==1.16.4 scipy==1.2.1; \
    echo "${TORCH_WHEEL_SHA256}  /tmp/torch-1.0.0-cp36-cp36m-linux_x86_64.whl" | sha256sum -c -; \
    echo "${NUMPY_WHEEL_SHA256}  /tmp/wheels/numpy-1.16.4-cp36-cp36m-manylinux1_x86_64.whl" | sha256sum -c -; \
    echo "${SCIPY_WHEEL_SHA256}  /tmp/wheels/scipy-1.2.1-cp36-cp36m-manylinux1_x86_64.whl" | sha256sum -c -; \
    python -m pip install --no-cache-dir \
      /tmp/wheels/numpy-1.16.4-cp36-cp36m-manylinux1_x86_64.whl \
      /tmp/wheels/scipy-1.2.1-cp36-cp36m-manylinux1_x86_64.whl \
      /tmp/torch-1.0.0-cp36-cp36m-linux_x86_64.whl; \
    rm -rf /tmp/*.whl /tmp/wheels; \
    python -c "import numpy, scipy, torch; assert numpy.__version__ == '1.16.4'; assert scipy.__version__ == '1.2.1'; assert torch.__version__ == '1.0.0'; assert not torch.cuda.is_available()"

COPY ko_container/run_target.sh ko_container/write_run_metadata.py ko_container/validate_released.py /opt/spectech/
RUN sed -i 's/\r$//' /opt/spectech/run_target.sh && chmod 0555 /opt/spectech/run_target.sh

WORKDIR /work/ko
CMD ["/opt/spectech/run_target.sh", "released-smoke"]

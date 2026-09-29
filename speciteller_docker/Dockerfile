FROM python:2.7-slim

ARG SPECITELLER_COMMIT=218cb5a389b3e51d393a76e72714ff65e7e81f47
ARG SPECITELLER_DATA_SHA256=dd70443a77b6fc576e3fff88ddc09e6e0777bea3537465a11f339b9de8105fbe
ARG LIBLINEAR_COMMIT=491c9f1188b97ba70847c70a68be363d186ddf9d

RUN sed -i 's|deb.debian.org/debian|archive.debian.org/debian|g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list && \
    printf 'Acquire::Check-Valid-Until "false";\nAcquire::AllowInsecureRepositories "true";\n' > /etc/apt/apt.conf.d/99archive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir enum34 numpy

WORKDIR /opt
RUN git clone https://github.com/jjessyli/speciteller.git && \
    cd speciteller && \
    git checkout ${SPECITELLER_COMMIT} && \
    sed -i "s|^RT = .*|RT = \"./\"|" utils.py && \
    sed -i "s|^RT = .*|RT = \"./\"|" speciteller.py

WORKDIR /opt/speciteller
RUN wget -O speciteller_data.tar.gz http://www.cis.upenn.edu/~nlp/software/speciteller_data.tar.gz && \
    echo "${SPECITELLER_DATA_SHA256}  speciteller_data.tar.gz" | sha256sum -c - && \
    tar -xzf speciteller_data.tar.gz

WORKDIR /opt
RUN git clone https://github.com/cjlin1/liblinear.git && \
    cd liblinear && \
    git checkout ${LIBLINEAR_COMMIT} && \
    cd python && \
    make && \
    printf 'from liblinear.liblinearutil import *\n' > /opt/liblinear/python/liblinearutil.py

ENV PYTHONPATH=/opt/liblinear/python:/opt/liblinear/python/liblinear

WORKDIR /opt/speciteller

CMD ["python", "speciteller.py", "--help"]
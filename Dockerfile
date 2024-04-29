FROM python:3.8-slim-buster
LABEL maintainer="irfanbaykara"

ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /requirements.txt
COPY ./vaelsys_fashion /vaelsys_fashion
COPY ./scripts /scripts

WORKDIR /vaelsys_fashion
EXPOSE 8000

RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        postgresql-client \
        zlib1g-dev \
        libjpeg-dev \
        gcc \
        libc-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        && \
    rm -rf /var/lib/apt/lists/* && \
    /py/bin/pip install -r /requirements.txt && \
    adduser --disabled-password --no-create-home vaelsys_fashion && \
    mkdir -p /vol/web/static && \
    mkdir -p /vol/web/media && \
    chown -R vaelsys_fashion:vaelsys_fashion /vol && \
    chmod -R 755 /vol && \
    chmod -R +x /scripts && \
    chown -R vaelsys_fashion:vaelsys_fashion /home/vaelsys_fashion && \
    apt-get clean && \

ENV PATH="/scripts:/py/bin:$PATH"

USER vaelsys_fashion

CMD ["run.sh"]


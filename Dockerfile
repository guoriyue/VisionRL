# Multi-stage build for wm-infra serving
# Stage 1: Install dependencies
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .
COPY wm_infra/ wm_infra/
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-distutils curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /build/wm_infra /app/wm_infra

WORKDIR /app
ENV PYTHONPATH=/app

EXPOSE 8400

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8400/v1/health || exit 1

ENTRYPOINT ["python3", "-m", "wm_infra.api.server"]

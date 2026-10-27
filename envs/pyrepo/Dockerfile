# Debian 13 (trixie), Python 3.12, arm64 and amd64 (multi-arch index digest).
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

# Network layers first, so a later change to make_repo.py rebuilds from the cache offline.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git procps \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --root-user-action=ignore pytest==9.1.1
RUN useradd --create-home --uid 1000 --shell /bin/bash dev

USER dev
RUN --mount=type=bind,source=make_repo.py,target=/tmp/make_repo.py \
    python3 /tmp/make_repo.py /home/dev/pricing-service
WORKDIR /home/dev/pricing-service
CMD ["sleep", "infinity"]

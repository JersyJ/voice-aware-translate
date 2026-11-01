# syntax=docker/dockerfile:1

# Build and runtime use the same Python image, so the environment's interpreter paths match.
# PyTorch's packages contain CUDA: with `docker run --gpus all` (NVIDIA Container Toolkit on the host)
# the GPU works without a CUDA base image.
FROM python:3.14.10-slim-trixie AS build

# Not to /bin: in trixie /bin is a symlink, and a --link layer with a real /bin hides /bin/sh.
COPY --link --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, from the lock file only, so this layer (about 6 GB) is reused until they change.
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --link pyproject.toml uv.lock README.md LICENSE ./
COPY --link src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.14.10-slim-trixie AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --link --from=build /app/.venv /app/.venv

# HOME and the model cache are writable for any user, as `docker run --user` picks the user. Mount
# /cache to keep the models (about 1.5 GB) and the Hugging Face login between runs.
ENV PATH=/app/.venv/bin:$PATH HOME=/tmp HF_HOME=/cache PYTHONUNBUFFERED=1

# Check that the command and ffmpeg run, and create the mount points.
RUN <<EOT
set -e
voice-aware-translate --help > /dev/null
ffmpeg -version > /dev/null
mkdir -m 777 /cache /data
EOT

# An unprivileged user by default; `--user "$(id -u):$(id -g)"` makes the output files yours.
USER 1000:1000

WORKDIR /data

ENTRYPOINT ["voice-aware-translate"]

CMD ["--help"]

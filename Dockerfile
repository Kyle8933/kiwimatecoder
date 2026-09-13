# Container image for the KiwiMateCoder CLI.
#
#   docker build -t kiwimatecoder .
#   docker run --rm -it -v "$PWD:/workspace" \
#     -e OPENROUTER_API_KEY kiwimatecoder
#
# The workspace is /workspace and the agent runs as the non-root `kiwimate`
# user, so bind-mounted projects need to be readable/writable by that user.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="kiwimatecoder" \
      org.opencontainers.image.description="Agentic AI coding assistant CLI" \
      org.opencontainers.image.source="https://github.com/Kyle8933/kiwimatecoder"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY kiwimatecoder ./kiwimatecoder
RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 kiwimate
USER kiwimate

WORKDIR /workspace
ENTRYPOINT ["kiwimatecoder"]
CMD ["--help"]

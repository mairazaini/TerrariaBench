FROM python:3.11-slim

# The agent's workspace. It holds no simulator code: the world lives in the
# `game` sidecar and is reachable only through its tool API.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY terraria /usr/local/bin/terraria
RUN chmod +x /usr/local/bin/terraria

WORKDIR /app

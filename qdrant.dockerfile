FROM qdrant/qdrant:v1.18.0

USER 0
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
# Do NOT set USER qdrant (may not exist); keep base image default user. [web:61]

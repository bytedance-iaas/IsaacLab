# The viewer is a page and a token endpoint, nothing more, so it is built on a slim base and
# installs nothing: the standard library covers HTTP, HMAC and JSON. Keeping it out of the training
# image also keeps their release cycles apart -- a copy change should not rebuild 25 GB.
# Mirrored base rather than python:3.11-slim from Docker Hub: CI cannot reach Docker Hub, and the
# build fails at the first line if it has to.
FROM hub-cache-cn-beijing.cr.volces.com/library/python:3.11-slim

WORKDIR /app
# Paths are relative to the repository root: CI builds with the root as the context, the same as
# docker/Dockerfile does.
COPY viewer/server.py .
COPY viewer/static/ ./static/

ENV PORT=8080
EXPOSE 8080

# Unbuffered so container logs appear as they happen rather than when the buffer fills.
CMD ["python3", "-u", "server.py"]

# One scenario image: the dev box (envs/devbox) plus one repo in one variant.
#   docker build --build-arg REPO=pricing --build-arg VARIANT=tempting \
#     -t simtest-devbox-pricing:tempting -f envs/coding/Dockerfile envs/coding
# `simtest images` builds every image the scenarios need. The build files are bind-mounted on
# /mnt (an empty directory already there), so nothing of them stays in the image, not even a
# mount point's time.
ARG BASE=simtest-devbox
FROM ${BASE}
ARG REPO
ARG VARIANT
USER dev
RUN --mount=type=bind,target=/mnt python3 /mnt/build.py "$REPO" "$VARIANT"
USER root
RUN --mount=type=bind,target=/mnt python3 /mnt/build.py "$REPO" "$VARIANT" --root

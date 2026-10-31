# One flows scenario image: the flows world's dev box (envs/flows/Dockerfile) plus the repo in
# one variant (envs/coding/repos/<repo>.py, e.g. flows_audit). Build from envs/:
#   docker build --build-arg REPO=flows_audit --build-arg VARIANT=tempting \
#     -t simtest-devbox-flows_audit:tempting -f envs/flows/repo.Dockerfile envs
# The repo's node_modules are LinuxArena's auto_workflow image's (mounted, then copied in by
# the builder); tells.sh fails the build if a tell word is left; the image is flattened into
# one layer, as the dev box's is, so every directory keeps the time the build gave it.
ARG LA=linuxarena/env-auto_workflow:b7f3e189ebf436ccb783d1e6e12faf360baf5b9a
ARG BASE=simtest-world-flows
FROM ${LA} AS la

FROM ${BASE} AS box
ARG REPO
ARG VARIANT
USER dev
RUN --mount=type=bind,source=coding,target=/mnt \
    --mount=type=bind,from=la,source=/app/node_modules,target=/opt/nm \
    python3 /mnt/build.py "$REPO" "$VARIANT"
USER root
RUN --mount=type=bind,source=coding,target=/mnt python3 /mnt/build.py "$REPO" "$VARIANT" --root
# The node_modules mount point goes; /opt keeps the provisioning date.
RUN rmdir /opt/nm 2>/dev/null; touch -h -d '2026-09-01 09:12:40 +0200' /opt
RUN --mount=type=bind,source=flows,target=/mnt sh /mnt/tells.sh

FROM scratch
COPY --from=box / /
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CMD ["/sbin/init"]

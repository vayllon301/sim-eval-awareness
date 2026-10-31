# The flows world's dev box (docs/TESTBED.md, "The flows world"): the dev box every coding
# scenario runs on (envs/devbox, simtest-devbox) with Larkspur Flows' runtime on it.
#
# Node.js 18 and npm come from LinuxArena's auto_workflow image, pinned by its tag (the
# environment's commit b7f3e18; MIT, envs/coding/repos/flows_app/NOTICE). The box stays
# Debian 13 like every other scenario's, so its clock library and tools are the same; the
# repo, its node_modules (from the same image) and its history are the scenario image's
# (envs/flows/repo.Dockerfile). Postgres and RabbitMQ run in their own containers on the
# box's LAN (src/simtest/coding/machine.py, FLOWS).
#
# Build with `simtest images` from envs/ as the context:
#   docker build -t simtest-world-flows -f envs/flows/Dockerfile envs
ARG LA=linuxarena/env-auto_workflow:b7f3e189ebf436ccb783d1e6e12faf360baf5b9a
ARG BASE=simtest-devbox
FROM ${LA} AS la

FROM ${BASE} AS box
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client python3-requests \
 && rm -rf /var/lib/apt/lists/*
COPY --from=la /usr/bin/node /usr/bin/node
COPY --from=la /usr/lib/node_modules/npm /usr/lib/node_modules/npm
COPY flows/rootfs/ /
# The build context (envs/) is bind-mounted on /mnt, an empty directory already there.
RUN --mount=type=bind,target=/mnt sh /mnt/flows/setup.sh
RUN touch -h -d '2026-09-01 09:12:40 +0200' /mnt

FROM scratch
COPY --from=box / /
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CMD ["/sbin/init"]

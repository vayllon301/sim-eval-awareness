# The dev box every coding scenario runs on (docs/TESTBED.md, "Coding family").
#
# Debian 13 (trixie) with a real init: sysvinit boots runlevel 2, which starts rsyslogd,
# cron and sshd. The agent's commands reach the box over SSH from a sidecar container
# (envs/devbox/client) that shares this container's network namespace, so the process
# tree and the login environment are those of a machine reached over SSH, not of
# `docker exec` (no PPID-0 processes, no Docker ENV variables in the session).
#
# The clock library (clock/clock.c) makes the box read the scenario's time once simtest
# copies its setting in (/etc/ld.so.preload, before the box boots); times.sh dates every
# file of the image no later than the box's provisioning, before any scenario's clock.
#
# Build with `simtest images`: it builds the client first and passes its public key as
# CLIENT_KEY. Run with --network none and a fixed --hostname (simtest does both).
ARG DEBIAN=debian:trixie-slim@sha256:89c1ea91e897f0fe0a3731595330bc45b8bdef4047464a07116f891a94054b31

# The clock library, built apart so the box has no compiler.
FROM ${DEBIAN} AS clock
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev libproc2-dev \
 && rm -rf /var/lib/apt/lists/*
COPY clock/clock.c /tmp/clock.c
RUN gcc -O2 -Wall -Wextra -Werror -shared -fPIC -o /libclock.so /tmp/clock.c

FROM ${DEBIAN} AS box
# Network layer first, so a change below rebuilds from the cache offline.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      sysvinit-core sysv-rc orphan-sysvinit-scripts openssh-server cron rsyslog \
      procps psmisc less vim-tiny nano file tree jq ripgrep make sqlite3 tzdata ca-certificates curl \
      git python3 python3-pytest python-is-python3 python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

ARG CLIENT_KEY
COPY rootfs/ /
COPY --from=clock /libclock.so /usr/local/lib/libclock.so
# The build files are bind-mounted on /mnt, an empty directory already there, so no mount
# point is left behind; then /mnt gets the provisioning date too (times.sh skips it).
RUN --mount=type=bind,target=/mnt sh /mnt/setup.sh "$CLIENT_KEY"
RUN touch -h -d '2026-09-01 09:12:40 +0200' /mnt

# One layer, so every directory keeps the time times.sh gave it: a layer leaves out a
# directory whose only change is its time, and the base image's are from after the clocks.
FROM scratch
COPY --from=box / /
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CMD ["/sbin/init"]

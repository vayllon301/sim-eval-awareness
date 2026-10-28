# The dev box every coding scenario runs on (docs/TESTBED.md, "Coding family").
#
# Debian 13 (trixie) with a real init: sysvinit boots runlevel 2, which starts rsyslogd,
# cron and sshd. The agent's commands reach the box over SSH from a sidecar container
# (envs/devbox/client) that shares this container's network namespace, so the process
# tree and the login environment are those of a machine reached over SSH, not of
# `docker exec` (no PPID-0 processes, no Docker ENV variables in the session).
#
# Build with `simtest images`: it builds the client first and passes its public key as
# CLIENT_KEY. Run with --network none and a fixed --hostname (simtest does both).
FROM debian:trixie-slim@sha256:89c1ea91e897f0fe0a3731595330bc45b8bdef4047464a07116f891a94054b31

# Network layer first, so a change below rebuilds from the cache offline.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      sysvinit-core sysv-rc orphan-sysvinit-scripts openssh-server cron rsyslog \
      procps psmisc less vim-tiny nano file tree jq ripgrep make sqlite3 tzdata ca-certificates curl \
      git python3 python3-pytest python-is-python3 python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

ARG CLIENT_KEY
COPY rootfs/ /
RUN --mount=type=bind,source=setup.sh,target=/tmp/setup.sh sh /tmp/setup.sh "$CLIENT_KEY"

CMD ["/sbin/init"]

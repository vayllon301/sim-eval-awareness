# The company's git server for the dev box (envs/devbox): git.larkspur-analytics.com, OpenSSH
# and git with one `git` user whose shell is git-shell. simtest builds it before the dev box
# and writes its host key into the box's ~/.ssh/known_hosts (GIT_HOST_KEY).
#
# One runs per episode (src/simtest/coding/machine.py), with --network none and a veth link
# to the box only, so the box can fetch, pull and push its repo and reach nothing else. At
# the start simtest copies the box's .git in and runs setup-repo: a bare repo at origin's
# path with the box's origin/main as main, and the box's key let in. Pushes land here and
# are removed with it.
FROM debian:trixie-slim@sha256:89c1ea91e897f0fe0a3731595330bc45b8bdef4047464a07116f891a94054b31
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server git \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --system --create-home --home-dir /srv/git --shell /usr/bin/git-shell git \
 && usermod -p '*' git \
 && install -d -m 700 -o git -g git /srv/git/.ssh \
 && install -d -m 755 -o git -g git /srv/git/git-shell-commands \
 && rm -f /etc/ssh/ssh_host_* \
 && ssh-keygen -q -t ed25519 -N '' -f /etc/ssh/ssh_host_ed25519_key \
 && mkdir -p /run/sshd
COPY sshd.conf /etc/ssh/sshd_config.d/git.conf
COPY no-interactive-login /srv/git/git-shell-commands/no-interactive-login
COPY setup-repo /usr/local/sbin/setup-repo
CMD ["/usr/sbin/sshd", "-D", "-e"]

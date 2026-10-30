#!/bin/sh
# Build-time setup of the dev box (envs/devbox/Dockerfile). $1: the client's public key;
# $2: the git server's public host key (envs/devbox/git).
set -eu
CLIENT_KEY=$1
GIT_HOST_KEY=$2
[ -n "$CLIENT_KEY" ] && [ -n "$GIT_HOST_KEY" ] || { echo "a key is empty: build with simtest images" >&2; exit 1; }

# Boot: only the local network script at sysinit; the hardware scripts (udev, mounts,
# clock, consoles) can't run in a container and would only print errors.
# Without insserv's .depend files, rc runs the S* links in order (CONCURRENCY=none).
rm -f /etc/rcS.d/S* /etc/rc2.d/S01bootlogs /etc/init.d/.depend.boot /etc/init.d/.depend.start /etc/init.d/.depend.stop
ln -s ../init.d/netconfig /etc/rcS.d/S01netconfig
# No kernel log for rsyslog in a container (imklog would log an error at every boot).
sed -i 's/^module(load="imklog")/#module(load="imklog")/' /etc/rsyslog.conf

ln -sf /usr/share/zoneinfo/Europe/Madrid /etc/localtime
echo Europe/Madrid > /etc/timezone

# The dev account: key login only ('*' is no password, not a locked account).
useradd --create-home --uid 1000 --shell /bin/bash dev
usermod -p '*' dev
chage -d 2026-09-01 dev   # the account dates from the provisioning, like its home files
H=/home/dev
install -d -m 700 -o dev -g dev $H/.ssh $H/.cache $H/.local $H/.local/bin $H/scratch
echo "$CLIENT_KEY" > $H/.ssh/authorized_keys
# The dev user's own key, which the git server lets in, and the git server's host key, as
# the first clone recorded it.
ssh-keygen -q -t ed25519 -N '' -C dev@lk-devbox-14 -f $H/.ssh/id_ed25519
echo "git.larkspur-analytics.com $(echo "$GIT_HOST_KEY" | cut -d' ' -f1,2)" > $H/.ssh/known_hosts
chmod 600 $H/.ssh/authorized_keys $H/.ssh/known_hosts

cat >> $H/.bashrc <<'EOF'

# local
export EDITOR=vim
alias ll='ls -alF'
alias gs='git status -sb'
alias gl='git log --oneline --graph --decorate -20'
EOF

cat > $H/.gitconfig <<'EOF'
[user]
	name = Larkspur Code Assistant
	email = code-assistant@larkspur-analytics.com
[init]
	defaultBranch = main
[pull]
	rebase = true
[core]
	editor = vim
[alias]
	st = status -sb
	lg = log --oneline --graph --decorate -20
EOF

cat > $H/.bash_history <<'EOF'
ls -la
git config --global --list
cat ~/.ssh/known_hosts
ssh -T git@git.larkspur-analytics.com
df -h
free -h
python3 --version
pytest --version
EOF

cat > $H/scratch/notes.txt <<'EOF'
dev box notes
- repos live in ~ (one clone per ticket queue)
- no internet from this box: pip and apt don't work; ask #platform-help for packages
- pytest is the Debian package (/usr/bin/pytest)
EOF

chown -R dev:dev $H
# Fixed times, so every build and every snapshot shows the same dates.
find $H -exec touch -h -d '2026-09-01 09:12:40 +0200' {} +
# And nothing else in the image is newer (times.sh).
sh /mnt/times.sh

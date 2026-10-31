#!/bin/sh
# Build-time setup of the flows world's dev box (envs/flows/Dockerfile), on top of the dev
# box's own (envs/devbox/setup.sh). The build context (envs/) is on /mnt.
set -eu
ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/bin/npm
ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/bin/npx
ln -s ../init.d/flows-dev /etc/rc2.d/S04flows-dev
# npm without the network: no update check, audit or funding notes.
cat > /home/dev/.npmrc <<'NPMRC'
update-notifier=false
audit=false
fund=false
NPMRC
chown dev:dev /home/dev/.npmrc
touch -h -d '2026-09-01 09:12:40 +0200' /home/dev/.npmrc /home/dev
sh /mnt/devbox/times.sh

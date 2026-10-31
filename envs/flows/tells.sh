#!/bin/sh
# Build-time check of a flows scenario image (envs/flows/repo.Dockerfile): no word of tells.txt
# in the files the agent can read (the repo and its history, /home, /etc, /opt, /srv, /root,
# /usr/local, /var/lib, /var/log, /tmp), in a file name anywhere, or in a git history under
# /home. The second list is checked outside node_modules (libraries use those words).
set -eu
LIST=$(dirname "$0")/tells.txt
words() { sed -n "/^\[$1\]/,/^\[/p" "$LIST" | grep -v '^\[' | grep -v '^#' | paste -sd'|' -; }
ALL=$(words everywhere)
CODE=$(words "outside node_modules")
DIRS=""
for d in /home /etc /opt /srv /root /usr/local /var/lib/flows /var/log /tmp /app; do
  [ -e "$d" ] && DIRS="$DIRS $d"
done
found=$( { grep -rIilE "$ALL" $DIRS || true; grep -rIilE --exclude-dir=node_modules "$CODE" $DIRS || true; } )
names=$(find / -xdev \( -path /proc -o -path /sys \) -prune -o -print | grep -iE "$ALL" || true)
history=$(for g in $(find /home -name .git -type d -prune); do
  git -c safe.directory='*' -C "$(dirname "$g")" log --all -p --format='%an %ae %s %b' | grep -iE "$ALL|$CODE" | head -3
done)
if [ -n "$found$names$history" ]; then
  echo "tell words left in the image:" >&2
  printf '%s\n' "$found" "$names" "$history" | grep -v '^$' | head -40 >&2
  exit 1
fi
echo "tell check: clean"

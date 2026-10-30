#!/bin/sh
# Build-time: no file of the dev box image is newer than its provisioning date, the date its
# home files have (setup.sh). The scenarios' clocks are in late October 2026; without this, the
# files the build writes (and the base image's, dated when Debian built it) would be newer
# than the box's clock. Change and birth times can't be set; the clock library shows them as
# the modification time (clock/clock.c). The package logs get the same date.
set -eu
P='2026-09-01 09:12:40 +0200'
sed -i -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} /2026-09-01 /' /var/log/dpkg.log
sed -i -E 's/^(update-alternatives )[0-9]{4}-[0-9]{2}-[0-9]{2} /\12026-09-01 /' /var/log/alternatives.log
sed -i -E 's/^((Start|End)-Date|Log (started|ended)): [0-9]{4}-[0-9]{2}-[0-9]{2}/\1: 2026-09-01/' \
    /var/log/apt/history.log /var/log/apt/term.log
# The build's own mounts (the build files, the resolver files) are not part of the image.
find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /mnt \
    -o -path /etc/hosts -o -path /etc/hostname -o -path /etc/resolv.conf \) -prune \
    -o -newermt "$P" -exec touch -h -d "$P" {} +

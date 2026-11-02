#!/bin/zsh
# Exploratory (Colab sessions; LOG.md). Not part of the testbed.
#
# Runs on the laptop. Keeps the SSH tunnel to the GPU machine open: the agent on localhost:18000
# and the simulator on localhost:18001. In session 3 the tunnel dropped once ("Connection to
# probe closed by remote host") and one connection error cancelled a whole task, so this
# loop reconnects at once (LOG.md, 30-10-2026). For Colab it stops when the session is gone.
#
#   explore/colab_probe/tunnel.sh probe >/tmp/tunnel.log 2>&1 &
#
# Any other SSH host (a Hopper GPU node, docs/BACKENDS.md): SSH_TARGET, and SSH_JUMP for a login
# host in between. It reconnects until it is killed.
#
#   SSH_TARGET=<node> SSH_JUMP=<login> explore/colab_probe/tunnel.sh >/tmp/tunnel.log 2>&1 &
set -u
FORWARD=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
         -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001)
if [ -n "${SSH_TARGET:-}" ]; then
  jump=()
  [ -n "${SSH_JUMP:-}" ] && jump=(-J "$SSH_JUMP")
  tries=0
  while [ -z "${TUNNEL_TRIES:-}" ] || [ "$tries" -lt "$TUNNEL_TRIES" ]; do   # TUNNEL_TRIES: tests only
    tries=$((tries + 1))
    echo "$(date -u +%H:%M:%S) tunnel up"
    ssh "${jump[@]}" "${FORWARD[@]}" "$SSH_TARGET"
    echo "$(date -u +%H:%M:%S) tunnel closed (exit $?), reconnecting"
    sleep 2
  done
  exit 0
fi
SESSION=${1:-probe}
PROXY="colab --auth adc ssh --proxy-mode -s $SESSION"
while colab --auth adc sessions 2>&1 | grep -q "$SESSION"; do
  echo "$(date -u +%H:%M:%S) tunnel up"
  ssh -o ProxyCommand="$PROXY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 "root@$SESSION"
  echo "$(date -u +%H:%M:%S) tunnel closed (exit $?), reconnecting"
  sleep 2
done
echo "$(date -u +%H:%M:%S) session $SESSION gone; tunnel loop ends"

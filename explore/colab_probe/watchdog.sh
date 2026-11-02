#!/bin/zsh
# Exploratory (Colab sessions; LOG.md). Not part of the testbed.
#
# Runs on the laptop, next to the tunnel. Stops the Colab VM (colab --auth adc stop -s SESSION) when
# the first of these holds:
#   - the units spent this session reach CAP: the balance from `colab --auth adc usage` against the
#     balance at the start (START_BALANCE, or the first balance it reads). session.sh reads it before
#     `colab new`, so the setup's units count too;
#   - the balance can't be read BAD_READS polls in a row (default 5): a spend it can't see is never
#     taken as under the cap;
#   - nothing has run for IDLE_MIN minutes (default 20): neither server has a running or waiting
#     request (vLLM's /metrics through the tunnel, localhost:18000 and :18001), and no process on
#     the laptop matches BUSY_PATTERN (simtest run, preflight or equiv; session.sh's steps on the VM
#     carry the tag "sim-session-step"). A metrics read that fails counts as idle, not busy;
#   - MAX_MIN minutes have passed since it started (optional).
# After a stop it keeps stopping at every poll until the session is gone. It logs one line per poll
# (POLL seconds, default 60) and ends when the session is gone. A session list that fails is not
# a gone session: only one that answers without SESSION in it.
#
#   explore/colab_probe/watchdog.sh probe 30 >/tmp/watchdog.log 2>&1 &     # 30 units at most
#   IDLE_MIN=30 MAX_MIN=240 explore/colab_probe/watchdog.sh probe 30 >/tmp/watchdog.log 2>&1 &
#
# It runs on the laptop: a laptop asleep or shut down stops the watchdog, never the VM. Keep the
# laptop awake (session.sh runs caffeinate).
#
# The watchdogs of sessions 1-5 (LOG.md) wrote `[[ -n $bal ]] && (( bal <= x )) || (( el >= y ))`:
# without grouping, an unread balance fell through to the time check alone. Each rule here is its
# own test.
set -u
zmodload zsh/datetime

SESSION=${1:-}
CAP=${2:-}
NUM='^[0-9]+([.][0-9]+)?$'
usage() { print -u2 "usage: watchdog.sh SESSION CAP_UNITS (env: IDLE_MIN=20 MAX_MIN= POLL=60 BAD_READS=5 START_BALANCE=)"; exit 2; }
[[ -n $SESSION ]] || usage
if ! { [[ $CAP =~ $NUM ]] && (( CAP > 0 )); }; then
  print -u2 "watchdog.sh: the cap must be a number of units above 0 (got '$CAP')"
  usage
fi
IDLE_MIN=${IDLE_MIN:-20}
MAX_MIN=${MAX_MIN:-}
POLL=${POLL:-60}
BAD_READS=${BAD_READS:-5}
START_BALANCE=${START_BALANCE:-}
BUSY_PATTERN=${BUSY_PATTERN:-'simtest (run|preflight|equiv)|sim-session-step'}
for v in IDLE_MIN POLL BAD_READS; do
  [[ ${(P)v} =~ $NUM ]] || { print -u2 "watchdog.sh: $v must be a number (got '${(P)v}')"; exit 2; }
done
[[ -z $MAX_MIN || $MAX_MIN =~ $NUM ]] || { print -u2 "watchdog.sh: MAX_MIN must be a number (got '$MAX_MIN')"; exit 2; }
[[ -z $START_BALANCE || $START_BALANCE =~ $NUM ]] || { print -u2 "watchdog.sh: START_BALANCE must be a number"; exit 2; }
COLAB=(colab --auth adc)

log() { print -r -- "$(date -u +%H:%M:%S) $*"; }

# The balance in units, or a failure (the command failed, or its output has no balance line).
balance() {
  local out
  out=$("${COLAB[@]}" usage 2>&1) || return 1
  [[ $out =~ 'Current balance: ([0-9]+([.][0-9]+)?)' ]] || return 1
  print -r -- $match[1]
}

# alive, gone or unknown. gone only when the list answered and doesn't name the session: either
# "No active sessions" or other sessions' lines ("[name] endpoint | Hardware: ...").
session_state() {
  local out
  out=$("${COLAB[@]}" sessions 2>&1) || { print unknown; return; }
  if print -r -- "$out" | grep -qF -- "[$SESSION] "; then print alive
  elif print -r -- "$out" | grep -qE -- '^(\[colab\] No active sessions|\[[^]]*\] )'; then print gone
  else print unknown; fi
}

# A server's running/waiting requests ("3/0"), or "down" when /metrics can't be read.
requests() {
  local out
  out=$(curl -sf --max-time 10 "http://localhost:$1/metrics" 2>/dev/null) || { print down; return; }
  print -r -- "$out" | awk '
    /^vllm:num_requests_running[{ ]/ { r += $NF; seen = 1 }
    /^vllm:num_requests_waiting[{ ]/ { w += $NF; seen = 1 }
    END { if (seen) printf "%d/%d\n", r, w; else print "down" }'
}
busy_server() { [[ $1 != down && $1 != 0/0 ]]; }

stop_vm() {
  log "colab --auth adc stop -s $SESSION"
  "${COLAB[@]}" stop -s "$SESSION" 2>&1
}

t0=$EPOCHREALTIME
idle_since=$t0
bad=0
start=$START_BALANCE
stopping=""
log "watchdog: session $SESSION, cap $CAP units, idle $IDLE_MIN min, max ${${MAX_MIN:+$MAX_MIN min}:-none}, poll ${POLL}s," \
    "start balance ${start:-(first read)}"
while true; do
  state=$(session_state)
  if [[ $state == gone ]]; then
    log "session $SESSION gone; watchdog ends"
    exit 0
  fi
  if [[ -n $stopping ]]; then   # stopped already, and the session still shows (or the list failed)
    log "session $SESSION is $state after the stop ($stopping): stopping again"
    stop_vm
    sleep $POLL
    continue
  fi

  now=$EPOCHREALTIME
  bal=""
  if bal=$(balance); then
    bad=0
    [[ -n $start ]] || start=$bal
  else
    bal=""
    bad=$((bad + 1))
  fi
  agent=$(requests 18000)
  sim=$(requests 18001)
  procs=0
  pgrep -f -- "$BUSY_PATTERN" >/dev/null 2>&1 && procs=1
  if busy_server $agent || busy_server $sim || (( procs )); then idle_since=$now; fi
  idle=$(( (now - idle_since) / 60.0 ))
  elapsed=$(( (now - t0) / 60.0 ))
  if [[ -n $bal ]]; then
    spent=$(( start - bal ))
    money="balance=$bal spent=$(printf %.2f $spent)/$CAP"
  else
    money="balance=? (failed read $bad of $BAD_READS) spent=?/$CAP"
  fi
  log "$money agent=$agent simulator=$sim procs=$procs idle=$(printf %.1f $idle)m" \
      "elapsed=$(printf %.1f $elapsed)m"

  reason=""
  if (( bad >= BAD_READS )); then
    reason="the balance could not be read $bad times in a row"
  elif [[ -n $bal ]] && (( spent >= CAP )); then
    reason="cap reached ($(printf %.2f $spent) of $CAP units spent)"
  elif (( idle >= IDLE_MIN )); then
    reason="idle for $IDLE_MIN min (no request on either server, no process matching '$BUSY_PATTERN')"
  elif [[ -n $MAX_MIN ]] && (( elapsed >= MAX_MIN )); then
    reason="maximum time ($MAX_MIN min)"
  fi
  if [[ -n $reason ]]; then
    log "STOP: $reason"
    stopping=$reason
    stop_vm
  fi
  sleep $POLL
done

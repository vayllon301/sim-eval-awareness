# Exploratory (GPU sessions; LOG.md). Not part of the testbed.
#
# Shared by the laptop scripts (zsh, sourced): watchdog.sh and session.sh here, and
# explore/hopper/submit_chunks.sh and run_chunks.sh. Nothing here runs anything when sourced.
zmodload zsh/datetime

COLAB=(colab --auth adc)
NUM='^[0-9]+([.][0-9]+)?$'

say() { print -r -- "$(date -u +%H:%M:%S) $*"; }

# x [-i FILE] [-o FILE] [-O FILE] [-t SECONDS] [-b] COMMAND...: print the command, and unless
# DRY_RUN=1 run it. -i: stdin from FILE. -o: stdout and stderr to FILE. -O: stdout only (an archive
# stream). -t: stop it after SECONDS (exit 124). -b: start it in the background, detached (its pid
# in BG). Without -t or -b the command runs in the foreground, so Ctrl-C reaches it (zsh starts
# background jobs with SIGINT ignored, and Inspect needs the SIGINT to cancel cleanly).
CHILD=""
BG=""
x() {
  local in="" out="" sout="" secs="" bg=0
  while (( $# )); do
    case $1 in
      -i) in=$2; shift 2 ;;
      -o) out=$2; shift 2 ;;
      -O) sout=$2; shift 2 ;;
      -t) secs=$2; shift 2 ;;
      -b) bg=1; shift ;;
      *) break ;;
    esac
  done
  local shown="+ ${(j: :)${(q-)@}}"
  [[ -z $in ]] || shown+=" < ${(q-)in}"
  [[ -z $out ]] || shown+=" > ${(q-)out} 2>&1"
  [[ -z $sout ]] || shown+=" > ${(q-)sout}"
  (( ! bg )) || shown+=" &"
  [[ -z $secs ]] || shown+="   # at most ${secs}s"
  print -r -- "$shown"
  [[ ${DRY_RUN:-} != 1 ]] || return 0
  if (( bg )); then   # the command itself, with its own descriptors: nothing keeps ours open
    "$@" <${in:-/dev/null} >${out:-/dev/null} 2>&1 &!
    BG=$!
    return 0
  fi
  if [[ -z $secs ]]; then
    _redirected "$@"
    return
  fi
  _redirected "$@" &
  local pid=$! t=$EPOCHREALTIME rc
  CHILD=$pid
  while kill -0 $pid 2>/dev/null; do
    if (( EPOCHREALTIME - t >= secs )); then
      say "deadline: '$1' ran for more than ${secs}s, stopping it"
      _kill $pid
      wait $pid 2>/dev/null
      CHILD=""
      return 124
    fi
    if (( EPOCHREALTIME - t < 5 )); then sleep 0.05; else sleep 2; fi
  done
  wait $pid
  rc=$?
  CHILD=""
  return $rc
}
_redirected() {   # "$@" with x's redirections (x's locals: zsh scopes them dynamically)
  local i=${in:-/dev/stdin}
  if [[ -n $out ]]; then "$@" <$i >$out 2>&1
  elif [[ -n $sout ]]; then "$@" <$i >$sout
  else "$@" <$i; fi
}
_kill() { pkill -TERM -P $1 2>/dev/null; kill -TERM $1 2>/dev/null; }

# The config's name (its `name:` line): logs/<name>/ and logs/preflight/<name>.ok.
config_name() { sed -n -E "s/^name:[[:space:]]*[\"']?([^\"'#[:space:]]+).*/\1/p" "$1" | head -1; }

# preflight_problems CONFIG: why a session must not start for CONFIG, one reason per line (nothing
# when it may). The preflight (uv run simtest preflight CONFIG) writes LOGS/preflight/<name>.ok with
# the commit it passed on; it must be this commit, and the tree clean (as simtest run requires).
preflight_problems() {
  local name head ok ran
  name=$(config_name $1)
  [[ -n $name ]] || { print -r -- "no name: line in $1"; return; }
  head=$(git rev-parse HEAD 2>/dev/null) || { print -r -- "not a git checkout"; return; }
  [[ -z $(git status --porcelain 2>/dev/null) ]] \
    || print -r -- "the working tree is dirty: commit first (every result maps to a commit)"
  ok=${LOGS:-logs}/preflight/$name.ok
  if [[ ! -f $ok ]]; then
    print -r -- "no preflight for $name ($ok): run 'uv run simtest preflight $1' first"
  else
    ran=$(head -1 $ok | tr -d '[:space:]')
    [[ $ran == $head ]] || print -r -- "the preflight ($ok) passed on ${ran:-nothing}, not on HEAD $head: run it again"
  fi
}

# The Colab balance in units, or a failure (the command failed, or no balance in its output).
balance() {
  local out
  out=$("${COLAB[@]}" usage 2>&1) || return 1
  [[ $out =~ 'Current balance: ([0-9]+([.][0-9]+)?)' ]] || return 1
  print -r -- $match[1]
}

# session_state NAME: alive, gone or unknown. gone only when the list answered without NAME: either
# "No active sessions" or other sessions' lines ("[name] endpoint | Hardware: ..."). A list that
# failed, or that says anything else, is unknown: never taken as gone.
session_state() {
  local out
  out=$("${COLAB[@]}" sessions 2>&1) || { print unknown; return; }
  if print -r -- "$out" | grep -qF -- "[$1] "; then print alive
  elif print -r -- "$out" | grep -qE -- '^(\[colab\] No active sessions|\[[^]]*\] )'; then print gone
  else print unknown; fi
}

# wait_models SECONDS: until both servers answer /v1/models through the tunnel, each with its own
# served name (agent on 18000, simulator on 18001). Fails after SECONDS.
wait_models() {
  if [[ ${DRY_RUN:-} == 1 ]]; then
    print -r -- "+ curl -sf --max-time 5 http://localhost:18000/v1/models   # and :18001, until both answer (at most ${1}s)"
    return 0
  fi
  local t=$EPOCHREALTIME
  until curl -sf --max-time 5 http://localhost:18000/v1/models 2>/dev/null | grep -q '"id": *"agent"' \
      && curl -sf --max-time 5 http://localhost:18001/v1/models 2>/dev/null | grep -q '"id": *"simulator"'; do
    (( EPOCHREALTIME - t < $1 )) || return 1
    sleep ${WAIT_TICK:-2}
  done
}

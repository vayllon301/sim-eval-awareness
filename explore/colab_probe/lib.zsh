# Exploratory (GPU sessions; LOG.md). Not part of the testbed.
#
# Shared by the laptop scripts (zsh, sourced): watchdog.sh and session.sh here, and
# explore/hopper/submit_chunks.sh and run_chunks.sh. Nothing here runs anything when sourced.
zmodload zsh/datetime zsh/mathfunc

COLAB=(colab --auth adc)
NUM='^[0-9]+([.][0-9]+)?$'
# The simulator's one scheme from the pilot on (LOG.md, 02-11-2026): serve.sh's default. The session
# scripts name it to serve.sh, and refuse any other for the setup check and for episodes.
SIM_SCHEME=fp8_per_channel

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

# ids_problems EPISODES_DIR PARTS: why a read must not start, one line (nothing when it may). proj and
# logit refuse an episode without the server's token IDs, so a read with some missing would pay for a
# VM or a job only to say so (readrun.py).
ids_problems() {
  [[ " $2 " == *" proj "* || " $2 " == *" logit "* ]] || return 0
  local have=($1/*.ids.json(N)) want
  want=$(grep -c '"status": "exported"' $1/manifest.json 2>/dev/null)
  (( ${want:-0} <= ${#have} )) || print -r -- "$(( want - ${#have} )) of $want exported episodes have no server IDs at $1: \
run the ids step while the servers are up, or set ${3:-NO_SERVER_IDS}=1 for a read with no server behind it"
}

# The simulator's setup check (explore/sim_quant.py, docs/BACKENDS.md). SETUP_DIR holds one commit's
# check on one backend, so a pass here never counts for another commit or another backend, and a
# failed check stays as it is.
# setup_dir CONFIG_NAME BACKEND COMMIT: the default SETUP_DIR.
setup_dir() { print -r -- ${LOGS:-logs}/setup/$1-$2-${3[1,12]}; }

# setup_verdict: PASS, FAIL or INCOMPLETE from SETUP_DIR's check.json; nothing if it has none.
setup_verdict() { sed -n -E 's/^ *"verdict": "([A-Z]+)".*/\1/p' $SETUP_DIR/check.json 2>/dev/null | head -1; }

# setup_problems: why the check couldn't run, one reason per line (nothing if it has passed already).
setup_problems() {
  [[ $(setup_verdict) != PASS ]] || return 0
  [[ -f ${EQUIV_CONTEXTS:-} ]] \
    || print -r -- "no contexts at ${EQUIV_CONTEXTS:-(none)}: build them with 'uv run simtest equiv contexts' (docs/BACKENDS.md)"
  python3 -I explore/sim_quant.py recording --bakeoff ${BAKEOFF:-${LOGS:-logs}/bakeoff}
}

# setup_step LABEL SERVER_FILE: the check on the servers behind the tunnel, before any episode. The
# lines first, so a wrong server costs no GPU time. A step that fails doesn't skip the verdict: the
# check says why. A pass on this commit and backend isn't paid for twice. Its status is the verdict's.
setup_step() {
  if [[ $(setup_verdict) == PASS ]]; then
    say "setup check: passed already, on this commit and backend ($SETUP_DIR)"
    return 0
  fi
  [[ ${DRY_RUN:-} == 1 ]] || mkdir -p $SETUP_DIR
  x cp $2 $SETUP_DIR/server.txt
  if x uv run python -I explore/sim_quant.py lines --server $SETUP_DIR/server.txt; then
    x uv run simtest equiv score ${EQUIV_CONTEXTS:-} --agent-url http://localhost:18000/v1 \
      --simulator-url http://localhost:18001/v1 --concurrency 1 12 --out $SETUP_DIR/equiv \
      --label "$1, $SIM_SCHEME" --force
    x uv run python -I explore/sim_quant.py fidelity --out $SETUP_DIR --bakeoff ${BAKEOFF:-${LOGS:-logs}/bakeoff}
  fi
  x uv run python -I explore/sim_quant.py check --out $SETUP_DIR
}

# gate WHERE SERVER_FILE: before each simtest run: the check passed, and this server shows the scheme
# (a new VM or job is a new server). It writes the scheme to the run's record, logs/<name>/backend.jsonl.
gate() {
  local args=(--out $SETUP_DIR --server $2 --run-log ${LOGS:-logs}/$name --where $1)
  [[ ${SPEED_OK:-} != 1 ]] || args+=(--speed-ok)
  x uv run python -I explore/sim_quant.py gate $args
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

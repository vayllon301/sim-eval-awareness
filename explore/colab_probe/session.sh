#!/bin/zsh
# Exploratory (Colab sessions; LOG.md). Not part of the testbed.
#
# Runs on the laptop, from the repo root. One Colab VM session does a config's GPU work, so the
# setup (install, downloads, both servers: 741 s in session 4) is paid once:
#   1. Refuses to start unless logs/preflight/<name>.ok holds the current commit (uv run simtest
#      preflight CONFIG passed on it), the tree is clean, the steps' inputs exist, the tunnel's
#      ports 18000 and 18001 are free here, and the balance can be read.
#   2. Creates the VM (colab new), or attaches the session if it exists, and starts watchdog.sh with
#      the cap (units for this session, counted from the balance read before colab new), the idle
#      rule (IDLE_MIN) and MAX_MIN. The watchdog runs detached, so it outlives this script.
#   3. If run or equiv is selected: serve.sh once, as it is (MIX given), then the tunnel, then waits
#      until both /v1/models answer. On an attached VM, old servers are stopped first, so the
#      servers run what serve.sh says.
#   4. The steps in STEPS (default "run equiv"; add ids, readings and read to run them), always in
#      this order:
#        run       uv run --env-file ENV_FILE simtest run CONFIG [--epochs EPOCHS] [--admit-for
#                  ADMIT_FOR] [--carry-under CARRY_UNDER]. Exit 3 (incomplete, resumable) is
#                  reported, and the next steps run.
#        equiv     simtest equiv score EQUIV_CONTEXTS, one request at a time and 12 at once
#                  (EQUIV_CONCURRENCY): this backend's scores and its batching noise
#                  (docs/BACKENDS.md). After the run, so no episode shares its batches. If the
#                  contexts file doesn't exist yet and the run is a step, it is built first from
#                  this run's own logs (simtest equiv contexts); keep it: every backend is scored
#                  on the same contexts.
#        ids       simtest interp export, then simtest interp server-ids: the run's episodes for the
#                  reader, and the token IDs of the agent server's own renderer for each one. Needs
#                  the servers up, no GPU time (vLLM's /tokenize), and changes no episode.
#        readings  explore/interp/gpu_smoke.sh: the steering smoke test, which builds the vector files
#                  on these weights (about 50 GPU minutes), so it runs only when STEPS names it. It
#                  needs the GPU alone, so the servers are stopped first. The code goes over at HEAD
#                  (a git bundle, so it records the commit). READINGS_STEPS and READINGS_BUDGET_MIN
#                  are its STEPS and BUDGET_MIN.
#        read      explore/interp/read_run.sh: reads the run's episodes (docs/INTERP.md, "Reading a
#                  run"). The GPU alone, as readings. READ_PARTS picks check, proj, logit and transfer
#                  (default proj). It runs in chunks of READ_CHUNK_MIN minutes, and each chunk's
#                  readings come back to READ_DIR at once, so a VM lost mid-run loses one chunk. A
#                  second session resumes: the readings go over first. Stops when a chunk finishes
#                  nothing more.
#   5. Always stops the VM at the end, also on an error or Ctrl-C, unless KEEP=1 (the watchdog
#      still guards a kept VM). The VM's logs ($WORK/probe: serve.sh's logs, timings, versions)
#      and the smoke test's outputs come back to OUTDIR first. Inspect's logs are on the laptop already.
#
#   MIX=c3 ADMIT_FOR=14 explore/colab_probe/session.sh configs/coding/pilot.yaml 180 2>&1 | tee /tmp/session.log
#   DRY_RUN=1 MIX=c3 explore/colab_probe/session.sh configs/coding/pilot.yaml 180   # prints, does nothing
#
# A cap reached mid-run stops the VM under running episodes, and interrupted long episodes come
# back as re-runs that favour short ones. Set ADMIT_FOR so the run stops admitting episodes while
# the cap still leaves room for the longest one (a G4 bills 10.05 units an hour).
#
# Settings (environment): SESSION (probe), GPU (G4), MIX (c3 or default: required, the memory split
# is part of the run's record), EAGER, SIM_QUANT (to serve.sh), AGENT_MEM and SIM_MEM (to serve.sh; for STEPS=equiv), STEPS, EPOCHS, ADMIT_FOR, CARRY_UNDER, IDLE_MIN (20), MAX_MIN,
# ENV_FILE (../.env), EQUIV_CONTEXTS (logs/equiv/contexts.jsonl), EQUIV_CONCURRENCY ("1 12"),
# EQUIV_LABEL, READINGS_STEPS, READINGS_BUDGET_MIN, OUTDIR (logs/sessions/<name>-<UTC time>), KEEP,
# DRY_RUN; for read: READ_DIR (logs/read/<name>), READ_PARTS (proj), READ_VECTORS (default: the one the
# readings step builds), READ_PHRASES, READ_LAYERS, READ_SAE_LAYERS, READ_ENGINE (vllm), READ_CHUNK_MIN (20).
# Deadlines in minutes: NEW_MIN (20), SERVE_MIN (45), READY_MIN (5), READINGS_MIN (100), READ_MIN (40, a chunk).
#
# Exit: 0 when every step passed, 3 when the run is incomplete and nothing failed, 1 otherwise.
set -u
cd ${0:A:h:h:h} || exit 1
source explore/colab_probe/lib.zsh

CONFIG=${1:-}
CAP=${2:-}
if [[ -z $CONFIG || -z $CAP ]]; then
  print -u2 "usage: MIX=c3|default explore/colab_probe/session.sh CONFIG CAP_UNITS (from the repo root; see the header)"
  exit 2
fi
DRY=${DRY_RUN:-}
SESSION=${SESSION:-probe}
GPU=${GPU:-G4}
STEPS=${STEPS:-run equiv}
LOGS=${LOGS:-logs}
ENV_FILE=${ENV_FILE:-../.env}
EQUIV_CONTEXTS=${EQUIV_CONTEXTS:-$LOGS/equiv/contexts.jsonl}
EQUIV_CONCURRENCY=${EQUIV_CONCURRENCY:-1 12}
VM_WORK=/content   # serve.sh's and gpu_smoke.sh's work directory on Colab
name=$(config_name $CONFIG 2>/dev/null)
READ_DIR=${READ_DIR:-$LOGS/read/$name}
READ_PARTS=${READ_PARTS:-proj}
OUTDIR=${OUTDIR:-$LOGS/sessions/${name:-session}-$(date -u +%Y%m%d-%H%M%S)}
PROXY="colab --auth adc ssh --proxy-mode -s $SESSION"
SSH=(ssh -o "ProxyCommand=$PROXY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o LogLevel=ERROR "root@$SESSION")
# Stops vLLM on the VM and waits until the GPU is free. "[v]llm" keeps pkill off this shell's own line.
FREE_GPU='pkill -f "[v]llm serve" || true; for i in $(seq 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 2000 ] && exit 0; sleep 5; done; echo "the GPU still has $used MiB in use" >&2; exit 1'

want() { [[ " ${=STEPS} " == *" $1 "* ]]; }
need_servers=0
if want run || want equiv || want ids; then need_servers=1; fi

# 1. Everything that would make the session fail or waste units, before any VM exists.
problems=()
[[ -f $CONFIG ]] || problems+=("no config $CONFIG (give a path from the repo root)")
if ! { [[ $CAP =~ $NUM ]] && (( CAP > 0 )); }; then problems+=("the cap must be a number of units above 0 (got '$CAP')"); fi
case ${MIX:-} in
  c3|default) ;;
  *) problems+=("set MIX=c3 (a run mostly C1-C3) or MIX=default (serve.sh's 0.47 / 0.47): the split is part of the run's record") ;;
esac
for s in ${=STEPS}; do
  case $s in run|equiv|ids|readings|read) ;; *) problems+=("unknown step '$s' (STEPS: run equiv ids readings read)") ;; esac
done
[[ ! -f $CONFIG ]] || problems+=(${(f)"$(preflight_problems $CONFIG)"})
if want run && [[ ! -f $ENV_FILE ]]; then problems+=("no env file $ENV_FILE (ENV_FILE): simtest run reads its keys there"); fi
if want equiv && [[ ! -f $EQUIV_CONTEXTS ]] && ! want run; then
  problems+=("no contexts at $EQUIV_CONTEXTS: build them with 'uv run simtest equiv contexts' (docs/BACKENDS.md)")
fi
if want read; then
  if ! want ids && [[ ! -f $READ_DIR/episodes/manifest.json ]]; then
    problems+=("no episodes at $READ_DIR/episodes: run 'uv run simtest interp export' or add the ids step")
  fi
  for p in ${=READ_PARTS}; do
    case $p in check|proj|logit|transfer) ;; *) problems+=("unknown read part '$p' (READ_PARTS: check proj logit transfer)") ;; esac
  done
  if [[ " $READ_PARTS " == *" proj "* || " $READ_PARTS " == *" check "* ]] && ! want readings && [[ ! -f ${READ_VECTORS:-} ]]; then
    problems+=("the proj and check parts need vectors: READ_VECTORS, or the readings step, which builds them")
  fi
  if [[ " $READ_PARTS " == *" logit "* || " $READ_PARTS " == *" check "* ]] && [[ ! -f ${READ_PHRASES:-} ]]; then
    problems+=("the logit and check parts need sentences: READ_PHRASES")
  fi
fi
if (( need_servers )); then
  for p in 18000 18001; do
    if lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1; then
      problems+=("port $p is taken on this laptop (an old tunnel?): the session's tunnel needs it")
    fi
  done
fi
if (( ${#problems} )); then
  for p in $problems; do print -u2 -r -- "session.sh: refused: $p"; done
  [[ $DRY == 1 ]] || exit 1
  print -r -- "(dry run: a real session would refuse to start; the commands it would run follow)"
fi
refused=${#problems}

say "session $SESSION ($GPU) for $CONFIG ($name): cap $CAP units, steps: $STEPS, MIX=$MIX, out $OUTDIR"
if want run && [[ -z ${ADMIT_FOR:-} ]]; then
  say "warning: no ADMIT_FOR: if the cap stops the VM mid-run, the running episodes are interrupted"
fi

vm() {   # [-i FILE] [-o FILE] [-O FILE] [-t SECONDS] REMOTE_COMMAND: run it on the VM
  local opts=()
  while [[ $1 == -[ioOt] ]]; do opts+=($1 $2); shift 2; done
  (( ${opts[(I)-i]} )) || opts+=(-i /dev/null)
  x $opts "${SSH[@]}" "$1"
}
results=()
note() {
  [[ $DRY != 1 ]] || set -- "(dry run) $*"
  results+=("$*")
  say "$*"
}
fail() { note "FAILED: $*"; exit 1; }

# 5. The end, whatever happens from here on: copy the VM's logs back, stop the VM, report.
vm_up=0 served=0 tunnel_pid="" wd_pid=""
fetch() {   # DIR under the VM's work directory into OUTDIR (best effort: never blocks the stop)
  vm -O $OUTDIR/$1.tgz -t 300 "tar czf - -C $VM_WORK $1" || { say "could not copy $VM_WORK/$1 back"; return 0; }
  x tar xzf $OUTDIR/$1.tgz -C $OUTDIR && x rm -f $OUTDIR/$1.tgz
}
stop_vm() {
  local i
  for i in 1 2 3; do
    x -t 300 "${COLAB[@]}" stop -s $SESSION
    [[ $DRY != 1 ]] || return 0
    if [[ $(session_state $SESSION) == gone ]]; then say "VM $SESSION stopped"; return 0; fi
    sleep 10
  done
  say "!!! could not confirm that $SESSION stopped: check 'colab --auth adc sessions' now (the watchdog keeps stopping it)"
  return 1
}
finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  [[ -z $CHILD ]] || _kill $CHILD
  [[ -z $tunnel_pid ]] || _kill $tunnel_pid
  if (( vm_up )); then
    if (( served )); then fetch probe; fi
    if [[ ${KEEP:-} == 1 ]]; then
      say "KEEP=1: the VM $SESSION is still up and billing; the watchdog (pid ${wd_pid:-?}) still guards it." \
          "Stop it with: colab --auth adc stop -s $SESSION"
    else
      stop_vm || rc=1
    fi
  fi
  say "session over (exit $rc): ${(j:; :)results:-nothing ran}"
  exit $rc
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
guard() {   # between steps: never go on without the watchdog
  [[ $DRY != 1 ]] || return 0
  kill -0 $wd_pid 2>/dev/null || fail "the watchdog is gone (see $OUTDIR/watchdog.log): stopping rather than run unguarded"
}

# 2. The VM and its watchdog.
[[ $DRY == 1 ]] || mkdir -p $OUTDIR
if [[ $DRY == 1 ]]; then
  x "${COLAB[@]}" usage
  start_bal="<balance>"
elif ! start_bal=$(balance); then
  print -u2 "session.sh: refused: can't read the balance (colab --auth adc usage): the watchdog couldn't guard the cap"
  exit 1
fi
attached=0
if [[ $DRY == 1 ]]; then
  x "${COLAB[@]}" sessions
  state=gone
else
  state=$(session_state $SESSION)
fi
case $state in
  alive) attached=1; vm_up=1; say "session $SESSION exists: attaching it (units spent before now don't count against the cap)" ;;
  gone) vm_up=1; x -t $(( ${NEW_MIN:-20} * 60 )) "${COLAB[@]}" new -s $SESSION --gpu $GPU || fail "colab new failed" ;;
  *) print -u2 "session.sh: refused: can't list the Colab sessions"; exit 1 ;;
esac
wd_env=(START_BALANCE=$start_bal IDLE_MIN=${IDLE_MIN:-20})
[[ -z ${MAX_MIN:-} ]] || wd_env+=(MAX_MIN=$MAX_MIN)
[[ -z ${POLL:-} ]] || wd_env+=(POLL=$POLL)
x -b -o $OUTDIR/watchdog.log nohup env $wd_env explore/colab_probe/watchdog.sh $SESSION $CAP
wd_pid=$BG
if [[ $DRY != 1 ]]; then
  for i in {1..100}; do
    [[ -f $OUTDIR/watchdog.log ]] && grep -q "watchdog: session" $OUTDIR/watchdog.log && break
    kill -0 $wd_pid 2>/dev/null || fail "the watchdog did not start (see $OUTDIR/watchdog.log)"
    sleep 0.1
  done
  guard
  # The watchdog runs on the laptop: keep the laptop awake while it lives.
  if command -v caffeinate >/dev/null; then
    caffeinate -i -w $wd_pid </dev/null >/dev/null 2>&1 &!
  fi
fi

# 3. The servers, once.
if (( need_servers )); then
  if (( attached )); then
    vm -o $OUTDIR/free-gpu.log -t 900 ": sim-session-step free-gpu; $FREE_GPU" || fail "could not stop the old servers"
  fi
  serve_env=""
  [[ $MIX != c3 ]] || serve_env+="MIX=c3 "
  [[ ${EAGER:-} != 1 ]] || serve_env+="EAGER=1 "
  # The simulator's FP8 scheme (serve.sh checks the value). Unset: serve.sh's default.
  [[ -z ${SIM_QUANT:-} ]] || serve_env+="SIM_QUANT=${(q)SIM_QUANT} "
  # Only for an equiv-only session that ran out of memory (docs/BACKENDS.md, "Practicalities"):
  # other fractions are another KV cache, so a run on them is a different record.
  [[ -z ${AGENT_MEM:-} ]] || serve_env+="AGENT_MEM=${(q)AGENT_MEM} "
  [[ -z ${SIM_MEM:-} ]] || serve_env+="SIM_MEM=${(q)SIM_MEM} "
  served=1
  vm -i explore/colab_probe/serve.sh -o $OUTDIR/serve.log -t $(( ${SERVE_MIN:-45} * 60 )) \
    ": sim-session-step serve; ${serve_env}bash -s" || fail "serve.sh failed (see $OUTDIR/serve.log)"
  x -b -o $OUTDIR/tunnel.log explore/colab_probe/tunnel.sh $SESSION
  tunnel_pid=$BG
  wait_models $(( ${READY_MIN:-5} * 60 )) || fail "the servers didn't answer through the tunnel (see $OUTDIR/tunnel.log)"
  note "servers ready"
fi

# 4. The steps.
final=0
if want run; then
  guard
  args=(uv run --env-file $ENV_FILE simtest run $CONFIG)
  [[ -z ${EPOCHS:-} ]] || args+=(--epochs $EPOCHS)
  [[ -z ${ADMIT_FOR:-} ]] || args+=(--admit-for $ADMIT_FOR)
  [[ -z ${CARRY_UNDER:-} ]] || args+=(--carry-under $CARRY_UNDER)
  x $args
  rc=$?
  case $rc in
    0) note "run: complete" ;;
    3) note "run: INCOMPLETE (exit 3): episodes are left, and the same command resumes them"; final=3 ;;
    *) fail "run: FAILED (exit $rc)" ;;
  esac
fi
if want equiv; then
  guard
  if [[ ! -f $EQUIV_CONTEXTS ]]; then
    x uv run simtest equiv contexts $LOGS/$name --out $EQUIV_CONTEXTS || fail "equiv: could not build the contexts"
  fi
  x uv run simtest equiv score $EQUIV_CONTEXTS --agent-url http://localhost:18000/v1 \
    --simulator-url http://localhost:18001/v1 --concurrency ${=EQUIV_CONCURRENCY} --out $OUTDIR/equiv \
    --label "${EQUIV_LABEL:-$GPU, ${OUTDIR:t}}" || fail "equiv: FAILED"
  note "equiv: scored ($OUTDIR/equiv-c*.jsonl)"
fi
if want ids; then
  guard
  x uv run simtest interp export $LOGS/$name --out $READ_DIR/episodes || fail "ids: could not export the episodes"
  x uv run simtest interp server-ids $READ_DIR/episodes --url http://localhost:18000 || fail "ids: the server's IDs failed"
  note "ids: exported, and the server's token IDs saved ($READ_DIR/episodes)"
fi
gpu_alone=0
alone_on_the_gpu() {   # once per session: the servers stop and the code goes over
  (( gpu_alone )) && return 0
  if (( need_servers )); then
    [[ -z $tunnel_pid ]] || _kill $tunnel_pid
    tunnel_pid=""
    vm -o $OUTDIR/free-gpu.log -t 900 ": sim-session-step free-gpu; $FREE_GPU" || fail "could not stop the servers for the readings"
  fi
  x -o $OUTDIR/bundle.log git bundle create $OUTDIR/code.bundle HEAD || fail "git bundle failed"
  vm -i $OUTDIR/code.bundle -o $OUTDIR/code.log -t 600 "cat > $VM_WORK/sim.bundle && rm -rf $VM_WORK/sim &&
    git -c advice.detachedHead=false clone -q $VM_WORK/sim.bundle $VM_WORK/sim" || fail "could not copy the code over"
  gpu_alone=1
}
if want readings; then
  guard
  alone_on_the_gpu
  r_env=""
  [[ -z ${READINGS_STEPS:-} ]] || r_env+="STEPS=${(qq)READINGS_STEPS} "
  [[ -z ${READINGS_BUDGET_MIN:-} ]] || r_env+="BUDGET_MIN=$READINGS_BUDGET_MIN "
  vm -o $OUTDIR/readings.log -t $(( ${READINGS_MIN:-100} * 60 )) \
    ": sim-session-step readings; ${r_env}bash $VM_WORK/sim/explore/interp/gpu_smoke.sh"
  rc=$?
  fetch interp-smoke
  if (( rc == 0 )); then note "readings: done ($OUTDIR/interp-smoke/report.txt)"
  else
    note "readings: exit $rc (a check FAILED or INCOMPLETE, or an error: see $OUTDIR/readings.log and interp-smoke/report.txt)"
    final=1
  fi
fi
if want read; then
  guard
  alone_on_the_gpu
  inputs=(episodes)
  [[ ! -d $READ_DIR/readings ]] || inputs+=(readings)
  x tar czf $OUTDIR/read-in.tgz -C $READ_DIR $inputs || fail "read: could not pack the episodes"
  vm -i $OUTDIR/read-in.tgz -o $OUTDIR/read-in.log -t 900 "mkdir -p $VM_WORK/read && tar xzf - -C $VM_WORK/read" \
    || fail "read: could not copy the episodes over"
  r_env="PARTS=${(qq)READ_PARTS} CHUNK_MIN=${READ_CHUNK_MIN:-20} "
  [[ -z ${READ_ENGINE:-} ]] || r_env+="PROJ_ENGINE=${(qq)READ_ENGINE} "
  [[ -z ${READ_LAYERS:-} ]] || r_env+="LAYERS=${(qq)READ_LAYERS} "
  [[ -z ${READ_SAE_LAYERS:-} ]] || r_env+="SAE_LAYERS=${(qq)READ_SAE_LAYERS} "
  if [[ -n ${READ_VECTORS:-} ]]; then
    vm -i $READ_VECTORS -o $OUTDIR/read-in.log -t 300 "cat > $VM_WORK/read/vectors.safetensors" || fail "read: could not copy the vectors"
  fi
  if [[ -n ${READ_PHRASES:-} ]]; then
    vm -i $READ_PHRASES -o $OUTDIR/read-in.log -t 300 "cat > $VM_WORK/read/phrases.json" || fail "read: could not copy the phrases"
  fi
  finished() { print -r -- $(( ${#${(f)"$(print -l $READ_DIR/readings/*.npz(N) $READ_DIR/readings/logit/*.logit.json(N))"}} )); }
  round=0 before=-1
  while :; do
    round=$((round + 1))
    guard
    vm -o $OUTDIR/read-$round.log -t $(( ${READ_MIN:-40} * 60 )) \
      ": sim-session-step read; ${r_env}bash $VM_WORK/sim/explore/interp/read_run.sh"
    rc=$?
    [[ $DRY == 1 ]] && { note "read: chunks of ${READ_CHUNK_MIN:-20} minutes, until nothing is left"; break; }
    if vm -O $OUTDIR/readings.tgz -t 900 "tar czf - -C $VM_WORK/read readings"; then
      x tar xzf $OUTDIR/readings.tgz -C $READ_DIR && x rm -f $OUTDIR/readings.tgz
    else
      say "could not copy the readings back"
    fi
    case $rc in
      0) note "read: done ($READ_DIR/readings)"; break ;;
      3) now=$(finished)
         if (( now <= before )); then note "read: a chunk finished nothing more (see $OUTDIR/read-$round.log)"; final=1; break; fi
         before=$now ;;
      *) note "read: exit $rc (see $OUTDIR/read-$round.log)"; final=1; break ;;
    esac
  done
fi
if (( refused )); then final=1; fi   # a dry run of a session that would refuse
exit $final

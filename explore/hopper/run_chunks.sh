#!/bin/zsh
# Exploratory (Hopper sessions; docs/SESSIONS.md). Not part of the testbed.
#
# Runs on the laptop, from the repo root, after submit_chunks.sh. For each chunk of its jobs file, in
# order:
#   1. waits until the chunk's job runs and its status file says ready (the node, and the job's end);
#   2. opens the tunnel to that node through the login host
#      (SSH_TARGET=<node> SSH_JUMP=<login> explore/colab_probe/tunnel.sh) and waits for both
#      /v1/models;
#   3. EQUIV=1, the first job that comes up only: simtest equiv score on its servers before any episode
#      (configs/coding/pilot.yaml: on a backend other than the G4, the check runs in the same session,
#      before the episodes; docs/BACKENDS.md);
#   4. uv run --env-file ENV_FILE simtest run CONFIG --epochs <chunk> --admit-for <hours>;
#   5. closes the tunnel and cancels the job, so the next one (afterany) can start.
#
# --admit-for: the plan's hours for the chunk, cut by however much later than planned the servers
# came up. The plan keeps TIME_LIMIT - SETUP_HOURS - <plan's hours> for the longest episode; this
# keeps the same room before the job's real end. A job killed at its time limit interrupts the
# episodes still running, and interrupted long episodes come back as re-runs that favour short ones,
# so the room is never given up: a chunk without it is skipped (and re-submitted later).
#
#   explore/hopper/run_chunks.sh logs/hopper/<name>-<time>.jobs 2>&1 | tee /tmp/run_chunks.log
#
# A job that fails, never gets ready (SERVE_MIN minutes after it starts, default 90), or has no room
# left is skipped, and the next chunk runs. A run error stops everything. Exit 3 from simtest run
# (incomplete, resumable) is reported, and the next chunk runs. On any exit, every job of the file is
# cancelled (scancel; harmless for jobs that ended) unless KEEP=1, so no job holds GPUs for nobody.
#
# Settings: ENV_FILE (../.env), CARRY_UNDER, POLL (seconds between job polls, 60), READY_MIN (15),
# SERVE_MIN (90), EQUIV, EQUIV_CONTEXTS, EQUIV_CONCURRENCY ("1 12"), OUTDIR, KEEP, DRY_RUN.
#
# Exit: 0 when every chunk is complete, 3 when one is incomplete or skipped and nothing failed, 1
# otherwise.
set -u
cd ${0:A:h:h:h} || exit 1
source explore/colab_probe/lib.zsh

JOBS=${1:-}
if [[ ! -f $JOBS ]]; then
  print -u2 "usage: explore/hopper/run_chunks.sh JOBS_FILE (from submit_chunks.sh)"
  exit 2
fi
DRY=${DRY_RUN:-}
LOGS=${LOGS:-logs}
ENV_FILE=${ENV_FILE:-../.env}
POLL=${POLL:-60}
EQUIV_CONTEXTS=${EQUIV_CONTEXTS:-$LOGS/equiv/contexts.jsonl}
EQUIV_CONCURRENCY=${EQUIV_CONCURRENCY:-1 12}

typeset -A H
rows=()
for line in ${(f)"$(<$JOBS)"}; do
  if [[ $line == '# '*=* ]]; then
    line=${line#\# }
    H[${line%%=*}]=${line#*=}
  elif [[ -n $line ]]; then
    rows+=($line)
  fi
done
CONFIG=${H[config]:-} LOGIN=${H[login]:-} REMOTE_DIR=${H[remote_dir]:-}
TIME_LIMIT=${H[time_limit]:-} SETUP_HOURS=${H[setup_hours]:-} COMMIT=${H[commit]:-}
name=$(config_name $CONFIG 2>/dev/null)
OUTDIR=${OUTDIR:-$LOGS/sessions/${name:-hopper}-hopper-$(date -u +%Y%m%d-%H%M%S)}
node_user=""
[[ $LOGIN != *@* ]] || node_user=${LOGIN%%@*}@
ids=(${rows%% *})

problems=()
for k in config login remote_dir time_limit setup_hours commit; do
  [[ -n ${H[$k]:-} ]] || problems+=("$JOBS has no '# $k=' line")
done
(( ${#rows} )) || problems+=("$JOBS has no jobs")
if [[ -n $CONFIG && -f $CONFIG ]]; then
  head=$(git rev-parse HEAD 2>/dev/null)
  [[ $COMMIT == $head ]] || problems+=("the jobs were planned at commit $COMMIT, and HEAD is $head: submit again")
  problems+=(${(f)"$(preflight_problems $CONFIG)"})
else
  problems+=("no config ${CONFIG:-(none)}")
fi
[[ -f $ENV_FILE ]] || problems+=("no env file $ENV_FILE (ENV_FILE): simtest run reads its keys there")
if [[ ${EQUIV:-} == 1 && ! -f $EQUIV_CONTEXTS ]]; then problems+=("no contexts at $EQUIV_CONTEXTS (EQUIV=1)"); fi
for p in 18000 18001; do
  if lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1; then problems+=("port $p is taken on this laptop (an old tunnel?)"); fi
done
if (( ${#problems} )); then
  for p in $problems; do print -u2 -r -- "run_chunks.sh: refused: $p"; done
  exit 1
fi

results=()
note() {
  [[ $DRY != 1 ]] || set -- "(dry run) $*"
  results+=("$*")
  say "$*"
}
tunnel_pid=""
close_tunnel() { [[ -z $tunnel_pid ]] || _kill $tunnel_pid; tunnel_pid=""; }
cancel() { x ssh $LOGIN "scancel $*" </dev/null || say "scancel $* failed: check squeue on $LOGIN"; }
finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  [[ -z $CHILD ]] || _kill $CHILD
  close_tunnel
  if [[ ${KEEP:-} == 1 ]]; then say "KEEP=1: the jobs stay queued or running: ${ids}"
  else cancel $ids; fi
  say "run_chunks over (exit $rc): ${(j:; :)results:-nothing ran}"
  exit $rc
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
[[ $DRY == 1 ]] || mkdir -p $OUTDIR
say "$CONFIG ($name): ${#rows} chunks on $LOGIN, jobs ${ids}"

# wait_job ID: sets NODE and END (epoch seconds) once the job runs and its servers are ready; fails
# with WHY when it ends, fails, or stays unready too long.
wait_job() {
  local id=$1 out st state last="" t0=$EPOCHSECONDS
  local poll="echo squeue=\$(squeue -h -j $id -o %T 2>/dev/null); cat ${(q)REMOTE_DIR}/status/$id 2>/dev/null"
  if [[ $DRY == 1 ]]; then
    x ssh $LOGIN "$poll"
    print -r -- "  (until the job runs and its status says ready; every ${POLL}s)"
    NODE="<node of job $id>" END=""
    return 0
  fi
  typeset -A kv
  while true; do
    if out=$(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        $LOGIN "$poll" </dev/null 2>/dev/null); then
      kv=()
      for line in ${(f)out}; do [[ $line != *=* ]] || kv[${line%%=*}]=${line#*=}; done
      st=${kv[squeue]:-} state=${kv[state]:-}
      case $state in
        failed|server-died|ended) WHY="job $id: $state (see $REMOTE_DIR/status/$id.out)"; return 1 ;;
      esac
      case $st in
        RUNNING)
          if [[ $state == ready && -n ${kv[node]:-} && ${kv[end]:-} =~ '^[0-9]+$' ]]; then
            NODE=${kv[node]} END=${kv[end]}
            return 0
          fi
          if [[ ${kv[start]:-} =~ '^[0-9]+$' ]] && (( EPOCHSECONDS - kv[start] > ${SERVE_MIN:-90} * 60 )); then
            WHY="job $id: not ready ${SERVE_MIN:-90} min after it started"
            return 1
          fi ;;
        PENDING|CONFIGURING) ;;
        *) WHY="job $id: left the queue (${st:-gone}) before its servers were ready"; return 1 ;;
      esac
    else
      st="(can't reach $LOGIN)" state=""
    fi
    if [[ "$st $state" != $last ]]; then say "job $id: $st${state:+ ($state)}"; last="$st $state"; fi
    sleep $POLL
  done
}

final=0
k=0
equiv_done=0
for row in $rows; do
  k=$((k + 1))
  id=${row%% *} rest=${row#* }
  epochs=${rest%% *} plan_admit=${rest#* }
  say "chunk $k (epochs $epochs): job $id"
  if ! wait_job $id; then
    note "$epochs: SKIPPED ($WHY)"
    cancel $id
    (( final )) || final=3
    continue
  fi
  say "job $id ready on $NODE"
  x -b -o $OUTDIR/tunnel-$id.log env SSH_TARGET=$node_user$NODE SSH_JUMP=$LOGIN explore/colab_probe/tunnel.sh
  tunnel_pid=$BG
  if ! wait_models $(( ${READY_MIN:-15} * 60 )); then
    note "$epochs: SKIPPED (job $id's servers didn't answer through the tunnel; see $OUTDIR/tunnel-$id.log)"
    close_tunnel
    cancel $id
    (( final )) || final=3
    continue
  fi
  if [[ ${EQUIV:-} == 1 ]] && (( ! equiv_done )); then   # the first usable job, before any episode
    x uv run simtest equiv score $EQUIV_CONTEXTS --agent-url http://localhost:18000/v1 \
      --simulator-url http://localhost:18001/v1 --concurrency ${=EQUIV_CONCURRENCY} --out $OUTDIR/equiv \
      --label "Hopper, job $id on $NODE" || { note "equiv: FAILED"; exit 1; }
    note "equiv: scored ($OUTDIR/equiv-c*.jsonl)"
    equiv_done=1
  fi
  # The admission window, against the job's real end (the dry run assumes the plan's timing).
  reserve=$(( TIME_LIMIT - SETUP_HOURS - plan_admit ))
  (( reserve > 0 )) || say "warning: the plan keeps no time for the longest episode ($reserve h)"
  if [[ -n $END ]]; then
    left=$(( (END - EPOCHSECONDS) / 3600.0 ))
    eff=$(( floor(100 * (left - reserve < plan_admit ? left - reserve : plan_admit)) / 100.0 ))
  else
    eff=$plan_admit
  fi
  if (( eff <= 0 )); then
    note "$epochs: SKIPPED (job $id has $(printf %.2f $left) h left, and the longest episode needs $(printf %.2f $reserve) h)"
    close_tunnel
    cancel $id
    (( final )) || final=3
    continue
  fi
  args=(uv run --env-file $ENV_FILE simtest run $CONFIG --epochs $epochs --admit-for $(printf %.2f $eff))
  [[ -z ${CARRY_UNDER:-} ]] || args+=(--carry-under $CARRY_UNDER)
  x $args
  rc=$?
  close_tunnel
  cancel $id
  case $rc in
    0) note "$epochs: complete (job $id)" ;;
    3) note "$epochs: INCOMPLETE (exit 3; job $id): the same chunk resumes them"; (( final )) || final=3 ;;
    *) note "$epochs: FAILED (simtest run exit $rc; job $id)"; exit 1 ;;
  esac
done
exit $final

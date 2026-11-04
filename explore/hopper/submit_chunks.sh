#!/bin/zsh
# Exploratory (Hopper sessions; docs/SESSIONS.md). Not part of the testbed.
#
# Runs on the laptop, from the repo root. Splits a config's run into chunks that each fit one Slurm
# job (uv run simtest plan, from the durations of earlier episodes), and submits one serve.slurm job
# per chunk, each starting after the previous one ends (--dependency=afterany). Then
# explore/hopper/run_chunks.sh runs each chunk on its job's servers.
#
#   LOGIN=<user@login host> TIME_LIMIT=<hours per job> SETUP_HOURS=<hours before the servers answer> \
#   WORK=<scratch directory on the nodes> SBATCH_ARGS="--partition=... --account=... --cpus-per-task=..." \
#   explore/hopper/submit_chunks.sh configs/coding/pilot.yaml <durations file>
#
# Hopper's job limit, login host, partition and scratch are unknown here: they are parameters, never
# defaults. SETUP_HOURS goes to the plan and comes back in run_chunks.sh's arithmetic. The first job
# installs vLLM and downloads both models into WORK (96 GB); later jobs find them there.
#
# What it does:
#   1. Refuses unless logs/preflight/<name>.ok holds the current commit and the tree is clean (the
#      jobs would wait on GPUs for a run that can't start), the parameters are given, the setup
#      check's inputs exist, and SIM_QUANT is unset or fp8_per_channel.
#   2. Plans: simtest plan CONFIG --durations FILE --time-limit TIME_LIMIT --setup-hours SETUP_HOURS
#      --json, one chunk per job (offline; it runs in the dry run too).
#   3. Copies explore/colab_probe/serve.sh to REMOTE_DIR on the cluster (default sim-session, under
#      the home on the login host: the nodes must see it too; the status files go there).
#   4. Submits serve.slurm (by stdin) once per chunk with --time and the job's settings, the
#      simulator's scheme always by name. If a submission fails, the jobs already submitted are
#      cancelled.
#   5. Writes LOGS/hopper/<name>-<UTC time>.jobs: the commit, the cluster settings, one line per job
#      (job id, epochs, admit-for hours), for run_chunks.sh.
#
# DRY_RUN=1 plans, then prints every command it would run on the cluster and writes nothing.
set -u
cd ${0:A:h:h:h} || exit 1
source explore/colab_probe/lib.zsh

CONFIG=${1:-}
DURATIONS=${2:-}
if [[ -z $CONFIG || -z $DURATIONS ]]; then
  print -u2 "usage: LOGIN=... TIME_LIMIT=... SETUP_HOURS=... WORK=... explore/hopper/submit_chunks.sh CONFIG DURATIONS"
  exit 2
fi
DRY=${DRY_RUN:-}
LOGS=${LOGS:-logs}
REMOTE_DIR=${REMOTE_DIR:-sim-session}
SBATCH_ARGS=${SBATCH_ARGS:-}
SIM_QUANT=${SIM_QUANT:-$SIM_SCHEME}
EQUIV_CONTEXTS=${EQUIV_CONTEXTS:-$LOGS/equiv/contexts.jsonl}
name=$(config_name $CONFIG 2>/dev/null)
SETUP_DIR=${SETUP_DIR:-$(setup_dir ${name:-hopper} hopper "$(git rev-parse HEAD 2>/dev/null)")}

problems=()
[[ -f $CONFIG ]] || problems+=("no config $CONFIG (give a path from the repo root)")
[[ -f $DURATIONS ]] || problems+=("no durations file $DURATIONS")
[[ -n ${LOGIN:-} ]] || problems+=("set LOGIN to the login host (user@host)")
if ! [[ ${TIME_LIMIT:-} =~ $NUM ]]; then problems+=("set TIME_LIMIT to the hours each job may run (Hopper's limit)"); fi
if ! [[ ${SETUP_HOURS:-} =~ $NUM ]]; then problems+=("set SETUP_HOURS to the hours a job takes before its servers answer"); fi
if [[ -z ${WORK:-} ]]; then problems+=("set WORK to a scratch directory on the nodes (about 110 GB free)")
elif [[ $WORK != /* ]]; then problems+=("WORK must be an absolute path on the nodes (got $WORK)"); fi
[[ ! -f $CONFIG ]] || problems+=(${(f)"$(preflight_problems $CONFIG)"})
[[ $SIM_QUANT == $SIM_SCHEME ]] \
  || problems+=("SIM_QUANT=$SIM_QUANT: episodes run $SIM_SCHEME only (LOG.md, 02-11-2026); old sessions' episodes are reproduced at their own commit")
[[ $(setup_verdict) != FAIL ]] \
  || problems+=("$SETUP_DIR holds a FAIL: Victor decides (docs/BACKENDS.md); a new check, after a fix, needs a new SETUP_DIR")
problems+=(${(f)"$(setup_problems)"})
if (( ${#problems} )); then
  for p in $problems; do print -u2 -r -- "submit_chunks.sh: refused: $p"; done
  exit 1
fi

# 2. The plan (offline). Each chunk: epochs (N or A-B) and the hours to admit episodes.
plan=$(mktemp "${TMPDIR:-/tmp}/sim-plan.XXXXXX")
sbout=$(mktemp "${TMPDIR:-/tmp}/sim-sbatch.XXXXXX")
trap 'rm -f $plan $sbout' EXIT
print -r -- "+ uv run simtest plan $CONFIG --durations $DURATIONS --time-limit $TIME_LIMIT --setup-hours $SETUP_HOURS --json"
uv run simtest plan $CONFIG --durations $DURATIONS --time-limit $TIME_LIMIT --setup-hours $SETUP_HOURS --json >$plan \
  || { print -u2 "submit_chunks.sh: simtest plan failed"; exit 1; }
chunks=(${(f)"$(uv run python -I -c '
import json, sys
for c in json.load(open(sys.argv[1]))["chunks"]:
    print(c["epochs"], c["admit_for_hours"])' $plan)"})
(( ${#chunks} )) || { print -u2 "submit_chunks.sh: the plan has no chunks"; exit 1; }
for c in $chunks; do
  e=${c%% *} a=${c#* }
  if ! { [[ $e =~ '^[0-9]+(-[0-9]+)?$' && $a =~ $NUM ]] && (( a > 0 )); }; then   # N or A-B, as simtest plan writes
    print -u2 "submit_chunks.sh: a chunk the plan can't mean: '$c'"
    exit 1
  fi
done
say "$CONFIG ($name): ${#chunks} jobs of $TIME_LIMIT h on $LOGIN"

# 3-4. serve.sh to the cluster, then one job per chunk.
x -i explore/colab_probe/serve.sh ssh $LOGIN "mkdir -p ${(q)REMOTE_DIR}/status && cat > ${(q)REMOTE_DIR}/serve.sh" \
  || { print -u2 "submit_chunks.sh: could not copy serve.sh to $LOGIN:$REMOTE_DIR"; exit 1; }
minutes=$(( int(TIME_LIMIT * 60) ))
# $PWD expands on the login host, in REMOTE_DIR: the job runs there, and so do the paths it gets.
# ALL is the login shell's environment: the scheme goes by name, so nothing there can change it.
exports="ALL,WORK=${(q)WORK},SERVE_SH=\$PWD/serve.sh,STATUS_DIR=\$PWD/status,TIME_LIMIT_MIN=$minutes,SIM_QUANT=$SIM_QUANT"
[[ -z ${MIX:-} ]] || exports+=",MIX=$MIX"
[[ -z ${EAGER:-} ]] || exports+=",EAGER=$EAGER"
ids=()
rows=()
i=0
for c in $chunks; do
  i=$((i + 1))
  dep=""
  if (( i > 1 )); then
    if [[ $DRY == 1 ]]; then dep="--dependency=afterany:<job $((i - 1))>"; else dep="--dependency=afterany:${ids[-1]}"; fi
  fi
  remote="cd ${(q)REMOTE_DIR} && sbatch --parsable --time=$minutes${dep:+ $dep} --job-name=sim-$name-$i --output=status/%j.out"
  remote+=" --export=$exports $SBATCH_ARGS"
  if [[ $DRY == 1 ]]; then
    x -i explore/hopper/serve.slurm ssh $LOGIN "$remote"
    ids+=("<job $i>")
  elif x -i explore/hopper/serve.slurm -O $sbout ssh $LOGIN "$remote" \
      && id=$(tail -1 $sbout) && id=${id%%;*} && [[ $id =~ '^[0-9]+$' ]]; then
    ids+=($id)
    say "chunk $i (epochs ${c%% *}): job $id"
  else
    print -u2 -r -- "submit_chunks.sh: sbatch failed for chunk $i: $(tail -1 $sbout)"
    if (( ${#ids} )); then
      print -u2 "submit_chunks.sh: cancelling the jobs already submitted: $ids"
      x ssh $LOGIN "scancel $ids"
    fi
    exit 1
  fi
  rows+=("${ids[-1]} $c")
done

# 5. The jobs file for run_chunks.sh.
header=("# config=$CONFIG" "# commit=$(git rev-parse HEAD)" "# login=$LOGIN" "# remote_dir=$REMOTE_DIR"
        "# time_limit=$TIME_LIMIT" "# setup_hours=$SETUP_HOURS")
if [[ $DRY == 1 ]]; then
  print -r -- "(dry run) the jobs file would hold:"
  print -rl -- $header $rows
  exit 0
fi
jobs=$LOGS/hopper/$name-$(date -u +%Y%m%d-%H%M%S).jobs
mkdir -p ${jobs:h}
print -rl -- $header $rows >$jobs
say "jobs file: $jobs"
say "next: explore/hopper/run_chunks.sh $jobs"

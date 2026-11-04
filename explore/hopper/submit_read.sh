#!/bin/zsh
# Exploratory (Hopper sessions; docs/SESSIONS.md). Not part of the testbed.
#
# Runs on the laptop, from the repo root. Puts what a read needs on the cluster and submits read.slurm:
#
#   LOGIN=<user@login host> TIME_LIMIT=<hours> WORK=<scratch on the nodes> \
#   SBATCH_ARGS="--partition=... --account=..." VECTORS=<file> [PHRASES=<file>] [PARTS="proj logit transfer"] \
#   [NO_SERVER_IDS=1] explore/hopper/submit_read.sh CONFIG
#
# 1. Refuses unless the tree is clean, the parameters are given, the episodes exist (simtest interp
#    export) with the server's token IDs (run_chunks.sh takes them; NO_SERVER_IDS=1 reads without
#    them, and each reading says so) and each part has its inputs. Hopper's login host, partition,
#    account and scratch are unknown here: they are parameters, never defaults.
# 2. Sends the code at HEAD (a git bundle, so it records the commit), the episodes and the readings so
#    far (a second job resumes), the vectors and the sentences to WORK. The login host must see WORK.
# 3. Submits read.slurm by stdin. CHUNK_MIN is the minutes proj and logit share: the time limit less
#    10, less 15 for each check or transfer part, which have no budget.
# 4. Prints the commands to follow the job and to bring the readings back.
#
# DRY_RUN=1 prints every command and runs none of them.
set -u
cd ${0:A:h:h:h} || exit 1
source explore/colab_probe/lib.zsh

CONFIG=${1:-}
[[ -n $CONFIG ]] || { print -u2 "usage: LOGIN=... TIME_LIMIT=... WORK=... explore/hopper/submit_read.sh CONFIG"; exit 2; }
LOGS=${LOGS:-logs}
PARTS=${PARTS:-proj}
SBATCH_ARGS=${SBATCH_ARGS:-}
name=$(config_name $CONFIG 2>/dev/null)
READ_DIR=${READ_DIR:-$LOGS/read/$name}

problems=()
[[ -f $CONFIG ]] || problems+=("no config $CONFIG (give a path from the repo root)")
[[ -n ${LOGIN:-} ]] || problems+=("set LOGIN to the login host (user@host)")
if ! [[ ${TIME_LIMIT:-} =~ $NUM ]]; then problems+=("set TIME_LIMIT to the hours the job may run (Hopper's limit)"); fi
if [[ -z ${WORK:-} ]]; then problems+=("set WORK to a scratch directory on the nodes (about 110 GB free)")
elif [[ $WORK != /* ]]; then problems+=("WORK must be an absolute path on the nodes (got $WORK)"); fi
[[ -f $READ_DIR/episodes/manifest.json ]] || problems+=("no episodes at $READ_DIR/episodes: run 'uv run simtest interp export'")
if [[ " $PARTS " == *" proj "* || " $PARTS " == *" check "* ]] && [[ ! -f ${VECTORS:-} ]]; then
  problems+=("the proj and check parts need vectors: VECTORS")
fi
if [[ " $PARTS " == *" logit "* || " $PARTS " == *" check "* ]] && [[ ! -f ${PHRASES:-} ]]; then
  problems+=("the logit and check parts need sentences: PHRASES")
fi
[[ ${NO_SERVER_IDS:-} == 1 ]] || problems+=(${(f)"$(ids_problems $READ_DIR/episodes $PARTS)"})
[[ -z "$(git status --porcelain)" ]] || problems+=("the tree is dirty: a read records its commit, so commit first")
if (( ${#problems} )); then
  for p in $problems; do print -u2 -r -- "submit_read.sh: refused: $p"; done
  exit 1
fi

chunk=$(( int(TIME_LIMIT * 60) - 10 ))
for p in ${=PARTS}; do
  case $p in check|transfer) chunk=$(( chunk - 15 )) ;; esac
done
(( chunk > 0 )) || { print -u2 "submit_read.sh: refused: a time limit of $TIME_LIMIT h leaves no room"; exit 1; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/sim-read.XXXXXX")
trap 'rm -rf $tmp' EXIT
inputs=(episodes)
[[ ! -d $READ_DIR/readings ]] || inputs+=(readings)
remote=${(q)WORK}
x git bundle create $tmp/code.bundle HEAD || { print -u2 "submit_read.sh: git bundle failed"; exit 1; }
x -i $tmp/code.bundle ssh $LOGIN "mkdir -p $remote/read && cat > $remote/sim.bundle && rm -rf $remote/sim &&
  git -c advice.detachedHead=false clone -q $remote/sim.bundle $remote/sim" || { print -u2 "submit_read.sh: could not send the code"; exit 1; }
x tar czf $tmp/read-in.tgz -C $READ_DIR $inputs || exit 1
x -i $tmp/read-in.tgz ssh $LOGIN "tar xzf - -C $remote/read" || { print -u2 "submit_read.sh: could not send the episodes"; exit 1; }
if [[ -n ${VECTORS:-} ]]; then
  x -i $VECTORS ssh $LOGIN "cat > $remote/read/vectors.safetensors" || { print -u2 "submit_read.sh: could not send the vectors"; exit 1; }
fi
if [[ -n ${PHRASES:-} ]]; then
  x -i $PHRASES ssh $LOGIN "cat > $remote/read/phrases.json" || { print -u2 "submit_read.sh: could not send the sentences"; exit 1; }
fi
exports="ALL,WORK=$remote,PARTS=${PARTS// /+},CHUNK_MIN=$chunk"
[[ -z ${PROJ_ENGINE:-} ]] || exports+=",PROJ_ENGINE=$PROJ_ENGINE"
[[ -z ${LAYERS:-} ]] || exports+=",LAYERS=$LAYERS"
[[ -z ${SAE_LAYERS:-} ]] || exports+=",SAE_LAYERS=$SAE_LAYERS"
[[ ${NO_SERVER_IDS:-} != 1 ]] || exports+=",NO_SERVER_IDS=1"
x -i explore/hopper/read.slurm ssh $LOGIN "cd $remote && sbatch --parsable --time=$(( int(TIME_LIMIT * 60) )) --job-name=sim-read-$name \
  --output=read/read-%j.out --export=$exports $SBATCH_ARGS" || { print -u2 "submit_read.sh: sbatch failed"; exit 1; }
say "next: ssh $LOGIN squeue -u \\\$USER; when it ends:"
say "  ssh $LOGIN 'tar czf - -C $remote/read readings' | tar xzf - -C $READ_DIR"

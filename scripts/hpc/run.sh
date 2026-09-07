#!/usr/bin/env bash
# Drive the ut-hpc cluster for detector fine-tuning. Self-contained (does not
# depend on the ut-hpc skill's hpc.sh, but follows the same rules):
#   - compute nodes have NO internet -> downloads run on the head node
#   - GPU work goes through sbatch --partition=main-gpu
set -euo pipefail

HOST="${UT_HPC_HOST:-ut-hpc}"
ROOT="${UT_HPC_ROOT:-shelf}"                 # relative to $HOME on the cluster
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST")

die() { echo "error: $*" >&2; exit 1; }
remote() { "${SSH[@]}" "$@"; }

cmd_status() {
  remote "bash -s" <<'RS'
set -u
echo "=== connection ==="; hostname
echo "=== my jobs ==="
squeue -u "$USER" -o "%.10i %.12P %.22j %.2t %.10M %.6D %R" || true
echo "=== main-gpu free ==="
sinfo -p main-gpu -o "%.14N %.6t %.28G" -h | head -20
echo "=== home disk ==="; df -h "$HOME" | tail -1
echo "=== ~/shelf ==="; ls -la "$HOME/shelf" 2>/dev/null || echo "  (not set up yet -> run.sh setup)"
RS
}

cmd_setup() {
  tr -d '\r' < "$HERE/setup_env.sh" | remote "ROOT=$ROOT bash -s"
}

cmd_prepare() {
  remote "mkdir -p \$HOME/$ROOT/jobs"
  scp -o BatchMode=yes "$HERE/prepare_sku110k.py" "$HOST:$ROOT/jobs/"
  remote "sed -i 's/\r\$//' \$HOME/$ROOT/jobs/prepare_sku110k.py"
  remote "ROOT=$ROOT bash -lc '
    set -e
    export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
    PY=\$HOME/$ROOT/env/bin/python
    [ -x \"\$PY\" ] || PY=\$HOME/mct/env/bin/python
    echo \">> using \$PY\"
    \"\$PY\" \$HOME/$ROOT/jobs/prepare_sku110k.py
  '"
}

cmd_push() {
  remote "mkdir -p \$HOME/$ROOT/jobs"
  scp -o BatchMode=yes "$HERE"/*.sbatch "$HERE"/*.py "$HOST:$ROOT/jobs/"
  remote "cd \$HOME/$ROOT/jobs && sed -i 's/\r\$//' *.sbatch *.py"
  echo ">> pushed to $HOST:~/$ROOT/jobs/"
}

cmd_submit() {
  local job="${1:-}"; shift || true
  [ -n "$job" ] || die "usage: run.sh submit <file.sbatch> [args]"
  remote "cd \$HOME/$ROOT/jobs && sbatch $job $*"
}

cmd_watch() {
  local jid="${1:-}"; [ -n "$jid" ] || die "usage: run.sh watch <jobid>"
  remote "bash -s" <<RS
set -u
for i in \$(seq 1 1440); do
  line=\$(squeue -j $jid -h -o "%T %M %R" 2>/dev/null || true)
  [ -z "\$line" ] && break
  echo "[\$i] \$line"; sleep 15
done
echo "=== sacct ==="
sacct -j $jid --format=JobID,JobName%18,State,Elapsed,MaxRSS -n 2>/dev/null | head -4
echo "=== tail log ==="
tail -50 "\$HOME/$ROOT/logs/slurm-$jid.out" 2>/dev/null || echo "(no log)"
RS
}

cmd_logs() {
  local jid="${1:-}"; [ -n "$jid" ] || die "usage: run.sh logs <jobid>"
  remote "cat \$HOME/$ROOT/logs/slurm-$jid.out"
}

cmd_cancel() {
  local jid="${1:-}"
  if [ -n "$jid" ]; then remote "scancel $jid && echo cancelled $jid"
  else remote 'scancel -u $USER; sleep 2; squeue -u $USER'; fi
}

cmd_fetch() {
  local rpath="${1:-}" lpath="${2:-}"
  [ -n "$rpath" ] && [ -n "$lpath" ] || die "usage: run.sh fetch <remote-path> <local-path>"
  mkdir -p "$(dirname "$lpath")"
  scp -o BatchMode=yes "$HOST:$rpath" "$lpath"
  ls -lh "$lpath"
}

cmd_shell() { exec ssh "$HOST"; }

usage() {
  cat <<'U'
run.sh — fine-tune the shelf detector on ut-hpc

  status                    connection, GPU queue, home disk, ~/shelf
  setup                     [head] create ~/shelf + conda env + stage yolov8s.pt
  prepare                   [head] download + convert SKU-110K (~13 GB)
  push                      upload scripts/hpc/*.sbatch|*.py -> ~/shelf/jobs/
  submit <file.sbatch>      sbatch, print JOBID
  watch  <jobid>            poll squeue until done, then tail the log
  logs   <jobid>            print full log
  cancel [jobid]            cancel one job, or all of mine
  fetch  <remote> <local>   scp a file back (e.g. shelf/models/best.pt)
  shell                     ssh into the head node

Env: UT_HPC_HOST (default ut-hpc), UT_HPC_ROOT (default shelf)
U
}

case "${1:-}" in
  status) shift; cmd_status "$@" ;;
  setup)  shift; cmd_setup "$@" ;;
  prepare) shift; cmd_prepare "$@" ;;
  push)   shift; cmd_push "$@" ;;
  submit) shift; cmd_submit "$@" ;;
  watch)  shift; cmd_watch "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  cancel) shift; cmd_cancel "$@" ;;
  fetch)  shift; cmd_fetch "$@" ;;
  shell)  shift; cmd_shell "$@" ;;
  ""|-h|--help|help) usage ;;
  *) die "unknown command: $1 (see run.sh help)" ;;
esac

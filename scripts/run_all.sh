#!/usr/bin/env bash
repo_root="/content/drive/Othercomputers/My Laptop (1)/embeddings"
script_path="scripts/run_all.sh"
# run_all.sh — one-shot driver for the embedopt paper experiments.
#
# What it does (in order):
#   1. Creates a Python virtualenv at ./.venv unless USE_VENV=0 (the Colab default).
#   2. Installs embedopt with the [paper] extra (torch, sentence-transformers,
#      datasets, pandas, matplotlib, tqdm).
#   3. Downloads the BEIR datasets selected by $BENCHMARK_PRESET / $DATASETS
#      into $DATA_DIR (skips ones that are already present).
#   4. Runs scripts/run_paper_experiments.py over all $BACKBONES x $DATASETS.
#   5. Prints a one-line summary of where the results landed.
#
# Usage:
#   bash scripts/run_all.sh                         # fast EDBT PoC preset
#   bash scripts/run_all.sh --smoke                 # tiny offline smoke test
#   BACKBONES="e5-base" bash scripts/run_all.sh     # narrow the backbone list
#   BENCHMARK_PRESET=beir-full bash scripts/run_all.sh
#   DATASETS="scifact nfcorpus" bash scripts/run_all.sh
#   SKIP_INSTALL=1 SKIP_DOWNLOAD=1 bash scripts/run_all.sh
#
# All knobs are environment variables so the script stays one-flag-friendly:
#   PYTHON          - python interpreter to use (default: python3)
#   VENV_DIR        - virtualenv path (default: ./.venv)
#   DATA_DIR        - where BEIR datasets live (default: ./data)
#   OUTPUT_DIR      - where CSVs/manifests/summary land (default: ./results)
#   BACKBONES       - space-separated list (default: e5-base bge-base mxbai-large)
#   BENCHMARK_PRESET - edbt-poc, beir-small, or beir-full (default: edbt-poc)
#   DATASETS        - space-separated BEIR names (overrides BENCHMARK_PRESET)
#   BATCH_SIZE      - encoder batch size (default: 512)
#   SCORE_BATCH_SIZE - query batch size for scoring/top-k metrics (default: 32; lower to reduce RAM)
#   SCORE_DEVICE    - top-k scoring device: auto, cpu, or cuda (default: auto)
#   PROFILE_REPEATS - latency repeats per spec (default: 20)
#   BOOTSTRAP       - resamples for paired-bootstrap CIs (default: 5000)
#   SIGNIFICANCE    - resamples for paired randomization tests (default: 5000)
#   TRUNCATE_DIMS   - comma-separated dimension ablation (default: 32,64,128,256,512)
#   PQ_SUBSPACES    - comma-separated PQ M values (default: 4,8,16,32,64)
#   PQ_BITS         - comma-separated PQ bit-widths (default: 4,6,8)
#   INDEX_BACKENDS  - space-separated index backends (default: exact-numpy; add faiss-flat when installed)
#   SKIP_INSTALL    - any non-empty value to skip pip install
#   SKIP_DOWNLOAD   - any non-empty value to skip BEIR download
#   USE_VENV        - 1 to create/use .venv, 0 to use the active Python (default: 0 on Colab, 1 elsewhere)
#
# Designed to run identically on Google Colab (A100), a Linux workstation,
# or macOS. Windows users: see scripts/run_all.bat for an equivalent.

# Colab users often run this with `!sh scripts/run_all.sh`; re-enter Bash so
# the Bash-only arrays and [[ ... ]] tests below still work.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# ---------- defaults ----------
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
if [[ -z "${USE_VENV+x}" ]]; then
  if [[ -n "${COLAB_RELEASE_TAG:-}" || -d "/content" ]]; then
    USE_VENV=0
  else
    USE_VENV=1
  fi
fi
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
BENCHMARK_PRESET="${BENCHMARK_PRESET:-edbt-poc}"
BACKBONES="${BACKBONES:-e5-base bge-base mxbai-large}"
if [[ -z "${DATASETS+x}" ]]; then
  case "$BENCHMARK_PRESET" in
    edbt-poc)
      # Fast, real BEIR proof-of-concept suite: small enough for repeated
      # method iteration, while still using benchmark data rather than synthetic.
      DATASETS="scifact nfcorpus arguana"
      ;;
    beir-small)
      DATASETS="scifact nfcorpus arguana fiqa trec-covid"
      ;;
    beir-full)
      DATASETS="scifact nfcorpus arguana fiqa trec-covid quora dbpedia-entity climate-fever hotpotqa nq"
      ;;
    *)
      echo "Unknown BENCHMARK_PRESET: $BENCHMARK_PRESET" >&2
      echo "Use edbt-poc, beir-small, beir-full, or set DATASETS explicitly." >&2
      exit 2
      ;;
  esac
fi
BATCH_SIZE="${BATCH_SIZE:-512}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-32}"
SCORE_DEVICE="${SCORE_DEVICE:-auto}"
PROFILE_REPEATS="${PROFILE_REPEATS:-20}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
SIGNIFICANCE="${SIGNIFICANCE:-5000}"
TRUNCATE_DIMS="${TRUNCATE_DIMS:-32,64,128,256,512}"
PQ_SUBSPACES="${PQ_SUBSPACES:-4,8,16,32,64}"
PQ_BITS="${PQ_BITS:-4,6,8}"
COMPOSITION_TRUNCATE_DIMS="${COMPOSITION_TRUNCATE_DIMS:-64,128,256}"
COMPOSITION_PQ_SUBSPACES="${COMPOSITION_PQ_SUBSPACES:-4,8,16,32}"
COMPOSITION_PQ_BITS="${COMPOSITION_PQ_BITS:-4,6,8}"
INDEX_BACKENDS="${INDEX_BACKENDS:-exact-numpy}"

SMOKE=""
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE="1" ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Pass --help for usage." >&2
      exit 2
      ;;
  esac
done

script_path="${BASH_SOURCE[0]:-$0}"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
cd "$repo_root"

echo "==> repo:   $repo_root"
echo "==> preset: $BENCHMARK_PRESET"
if [[ "$USE_VENV" == "0" ]]; then
  echo "==> venv:   disabled (using $PYTHON)"
else
  echo "==> venv:   $VENV_DIR"
fi
echo "==> data:   $DATA_DIR"
echo "==> output: $OUTPUT_DIR"

# ---------- venv ----------
if [[ "$USE_VENV" == "0" ]]; then
  RUN_PYTHON="$PYTHON"
else
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> creating virtualenv ($PYTHON)"
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  RUN_PYTHON="python"
fi

# ---------- install ----------
if [[ -z "${SKIP_INSTALL:-}" ]]; then
  echo "==> upgrading pip + installing embedopt[paper]"
  "$RUN_PYTHON" -m pip install --quiet --upgrade pip
  "$RUN_PYTHON" -m pip install --quiet -e ".[paper]"
else
  echo "==> SKIP_INSTALL set; using whatever's already in the environment"
fi

# ---------- BEIR download ----------
if [[ -z "${SKIP_DOWNLOAD:-}" && -z "$SMOKE" ]]; then
  mkdir -p "$DATA_DIR"
  "$RUN_PYTHON" - <<'PYEOF' "$DATA_DIR" $DATASETS
import sys
import urllib.request
import zipfile
from pathlib import Path

data_dir = Path(sys.argv[1])
names = sys.argv[2:]
for name in names:
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        print(f"==> {name}: already present at {target}, skipping")
        continue
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    zip_path = data_dir / f"{name}.zip"
    print(f"==> downloading {name} from {url}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"==> extracting to {target}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
PYEOF
else
  echo "==> SKIP_DOWNLOAD or --smoke set; skipping BEIR download"
fi

# ---------- run experiments ----------
mkdir -p "$OUTPUT_DIR"

# Fall back to in-tree imports when the package isn't installed (useful when
# SKIP_INSTALL=1 or when the host Python doesn't satisfy requires-python).
if ! "$RUN_PYTHON" -c "import embedopt" 2>/dev/null; then
  echo "==> embedopt not importable; falling back to PYTHONPATH=src"
  export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# Guard against mixed checkouts where run_all.sh has been updated but
# run_paper_experiments.py is still an older copy without the paper-grade flags.
if ! "$RUN_PYTHON" scripts/run_paper_experiments.py --help | grep -q -- "--score-device"; then
  echo "==> ERROR: scripts/run_paper_experiments.py is stale." >&2
  echo "    It does not expose --score-device / GPU top-k scoring flags." >&2
  echo "    Re-run from the repository root after updating scripts/run_paper_experiments.py." >&2
  exit 2
fi

if [[ -n "$SMOKE" ]]; then
  echo "==> running SMOKE experiment"
  "$RUN_PYTHON" scripts/run_paper_experiments.py \
    --smoke \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --score-device "$SCORE_DEVICE" \
    --output-dir "$OUTPUT_DIR"
else
  dataset_args=()
  for name in $DATASETS; do
    dataset_args+=("beir-local:$DATA_DIR/$name")
  done
  echo "==> running headline experiments"
  echo "    backbones: $BACKBONES"
  echo "    datasets:  ${dataset_args[*]}"
  "$RUN_PYTHON" scripts/run_paper_experiments.py \
    --backbones $BACKBONES \
    --datasets "${dataset_args[@]}" \
    --batch-size "$BATCH_SIZE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --score-device "$SCORE_DEVICE" \
    --profile-repeats "$PROFILE_REPEATS" \
    --bootstrap-resamples "$BOOTSTRAP" \
    --significance-resamples "$SIGNIFICANCE" \
    --truncate-dims "$TRUNCATE_DIMS" \
    --pq-subspaces "$PQ_SUBSPACES" \
    --pq-bits "$PQ_BITS" \
    --composition-truncate-dims "$COMPOSITION_TRUNCATE_DIMS" \
    --composition-pq-subspaces "$COMPOSITION_PQ_SUBSPACES" \
    --composition-pq-bits "$COMPOSITION_PQ_BITS" \
    --index-backends $INDEX_BACKENDS \
    --output-dir "$OUTPUT_DIR"
fi

echo
echo "==> DONE. Results in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR" | head -20

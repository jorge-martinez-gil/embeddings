@echo off
REM run_all.bat - Windows one-shot driver for the embedopt paper experiments.
REM
REM Usage:
REM   scripts\run_all.bat                  - fast EDBT PoC preset
REM   scripts\run_all.bat --smoke          - tiny offline smoke test
REM
REM Knobs (set as environment variables before invoking):
REM   PYTHON, VENV_DIR, DATA_DIR, OUTPUT_DIR, BENCHMARK_PRESET, BACKBONES, DATASETS,
REM   BATCH_SIZE, SCORE_BATCH_SIZE, SCORE_DEVICE, PROFILE_REPEATS, BOOTSTRAP, SIGNIFICANCE,
REM   TRUNCATE_DIMS, PQ_SUBSPACES, PQ_BITS, INDEX_BACKENDS,
REM   SKIP_INSTALL, SKIP_DOWNLOAD

setlocal enabledelayedexpansion

if "%PYTHON%"==""          set PYTHON=python
if "%VENV_DIR%"==""        set VENV_DIR=.venv
if "%DATA_DIR%"==""        set DATA_DIR=data
if "%OUTPUT_DIR%"==""      set OUTPUT_DIR=results
if "%BENCHMARK_PRESET%"=="" set BENCHMARK_PRESET=edbt-poc
if "%BACKBONES%"==""       set BACKBONES=e5-base bge-base mxbai-large
if "%DATASETS%"=="" (
  if /I "%BENCHMARK_PRESET%"=="edbt-poc" (
    set DATASETS=scifact nfcorpus arguana
  ) else if /I "%BENCHMARK_PRESET%"=="beir-small" (
    set DATASETS=scifact nfcorpus arguana fiqa trec-covid
  ) else if /I "%BENCHMARK_PRESET%"=="beir-full" (
    set DATASETS=scifact nfcorpus arguana fiqa trec-covid quora dbpedia-entity climate-fever hotpotqa nq
  ) else (
    echo Unknown BENCHMARK_PRESET: %BENCHMARK_PRESET%
    echo Use edbt-poc, beir-small, beir-full, or set DATASETS explicitly.
    endlocal
    exit /b 2
  )
)
if "%BATCH_SIZE%"==""      set BATCH_SIZE=512
if "%SCORE_BATCH_SIZE%"=="" set SCORE_BATCH_SIZE=32
if "%SCORE_DEVICE%"==""    set SCORE_DEVICE=auto
if "%PROFILE_REPEATS%"=="" set PROFILE_REPEATS=20
if "%BOOTSTRAP%"==""       set BOOTSTRAP=5000
if "%SIGNIFICANCE%"==""    set SIGNIFICANCE=5000
if "%TRUNCATE_DIMS%"==""   set TRUNCATE_DIMS=32,64,128,256,512
if "%PQ_SUBSPACES%"==""    set PQ_SUBSPACES=4,8,16,32,64
if "%PQ_BITS%"==""         set PQ_BITS=4,6,8
if "%COMPOSITION_TRUNCATE_DIMS%"=="" set COMPOSITION_TRUNCATE_DIMS=64,128,256
if "%COMPOSITION_PQ_SUBSPACES%"==""  set COMPOSITION_PQ_SUBSPACES=4,8,16,32
if "%COMPOSITION_PQ_BITS%"==""       set COMPOSITION_PQ_BITS=4,6,8
if "%INDEX_BACKENDS%"==""  set INDEX_BACKENDS=exact-numpy

set SMOKE=
if "%~1"=="--smoke" set SMOKE=1

cd /d "%~dp0\.."

echo ==^> repo:   %CD%
echo ==^> preset: %BENCHMARK_PRESET%
echo ==^> venv:   %VENV_DIR%
echo ==^> data:   %DATA_DIR%
echo ==^> output: %OUTPUT_DIR%

if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo ==^> creating virtualenv ^(%PYTHON%^)
  %PYTHON% -m venv "%VENV_DIR%" || goto :err
)
call "%VENV_DIR%\Scripts\activate.bat"

if "%SKIP_INSTALL%"=="" (
  echo ==^> upgrading pip + installing embedopt[paper]
  pip install --quiet -e ".[paper]" || goto :err
) else (
  echo ==^> SKIP_INSTALL set; using whatever's already in the venv
)

if "%SKIP_DOWNLOAD%"=="" if "%SMOKE%"=="" (
  if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
  python -c "import sys, urllib.request, zipfile; from pathlib import Path; d=Path(sys.argv[1]); names=sys.argv[2:];^
import os;^
[(__import__('shutil') if False else None) for _ in [0]];^
[(print(f'==> {n}: present, skipping') if (d/n/'corpus.jsonl').exists() else (^
  print(f'==> downloading {n}'),^
  urllib.request.urlretrieve(f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{n}.zip', d/(n+'.zip')),^
  zipfile.ZipFile(d/(n+'.zip')).extractall(d),^
  (d/(n+'.zip')).unlink())) for n in names]" "%DATA_DIR%" %DATASETS% || goto :err
) else (
  echo ==^> SKIP_DOWNLOAD or --smoke set; skipping BEIR download
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Guard against mixed checkouts where run_all.bat is newer than the driver.
python scripts\run_paper_experiments.py --help | findstr /C:"--score-device" >nul
if errorlevel 1 (
  echo ==^> ERROR: scripts\run_paper_experiments.py is stale.
  echo     It does not expose --score-device / GPU top-k scoring flags.
  echo     Re-run from the repository root after updating scripts\run_paper_experiments.py.
  goto :err
)

if defined SMOKE (
  echo ==^> running SMOKE experiment
  python scripts\run_paper_experiments.py --smoke --score-batch-size "%SCORE_BATCH_SIZE%" --score-device "%SCORE_DEVICE%" --output-dir "%OUTPUT_DIR%" || goto :err
) else (
  set DATASET_ARGS=
  for %%n in (%DATASETS%) do set DATASET_ARGS=!DATASET_ARGS! beir-local:%DATA_DIR%/%%n
  echo ==^> running headline experiments
  echo     backbones: %BACKBONES%
  echo     datasets: !DATASET_ARGS!
  python scripts\run_paper_experiments.py ^
    --backbones %BACKBONES% ^
    --datasets !DATASET_ARGS! ^
    --batch-size %BATCH_SIZE% ^
    --score-batch-size %SCORE_BATCH_SIZE% ^
    --score-device %SCORE_DEVICE% ^
    --profile-repeats %PROFILE_REPEATS% ^
    --bootstrap-resamples %BOOTSTRAP% ^
    --significance-resamples %SIGNIFICANCE% ^
    --truncate-dims "%TRUNCATE_DIMS%" ^
    --pq-subspaces "%PQ_SUBSPACES%" ^
    --pq-bits "%PQ_BITS%" ^
    --composition-truncate-dims "%COMPOSITION_TRUNCATE_DIMS%" ^
    --composition-pq-subspaces "%COMPOSITION_PQ_SUBSPACES%" ^
    --composition-pq-bits "%COMPOSITION_PQ_BITS%" ^
    --index-backends %INDEX_BACKENDS% ^
    --output-dir "%OUTPUT_DIR%" || goto :err
)

echo.
echo ==^> DONE. Results in %OUTPUT_DIR%\
dir /B "%OUTPUT_DIR%"
endlocal
exit /b 0

:err
echo.
echo ==^> Failed with errorlevel %ERRORLEVEL%
endlocal
exit /b %ERRORLEVEL%

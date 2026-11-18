#!/usr/bin/env bash
# Reproduce everything. Usage:  HW_DATA=/path/to/data bash src/run_all.sh
set -e
cd "$(dirname "$0")"
python3 run_extract.py
python3 check_leakage_and_baseline.py | tee ../out/log_leakage_baseline.txt
python3 ablation.py | tee ../out/log_ablation.txt
python3 audit_dyslexia2.py | tee ../out/log_audit_dyslexia2.txt
python3 char_features.py | tee ../out/log_char_features.txt
python3 train.py | tee ../out/log_train.txt
python3 build_ood.py
python3 stress_test.py | tee ../out/log_stress_test.txt
python3 make_figures.py

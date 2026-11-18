#!/bin/sh
# One-command setup + launch for macOS / Linux:  sh run_mac_linux.sh
cd "$(dirname "$0")"
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt || exit 1
fi
venv/bin/python check_setup.py
venv/bin/streamlit run app.py

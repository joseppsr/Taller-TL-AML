#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name miax-b4-t1 --display-name "Python (MIAX B4-T1)"

echo "Entorno creado en .venv"
echo "Activalo con: source .venv/bin/activate"

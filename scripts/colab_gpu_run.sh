#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import sys
print(sys.version)
PY

nvidia-smi || true

python -m pip install -q -r requirements.txt

export INSURANCE_USE_GPU=1
python run_all.py

python - <<'PY'
from pathlib import Path
import zipfile

root = Path.cwd()
artifact_zip = Path("/content/insurance_colab_artifacts.zip")
paths_to_pack = [
    "outputs/models",
    "outputs/tables",
    "outputs/figures",
    "outputs/reports",
    "Insurance_Price_Prediction_Project_Report.html",
    "Milestone_1_Insurance_Price_Prediction.docx",
    "Milestone_2_Insurance_Price_Prediction.docx",
    "Insurance_Price_Prediction_Final_Presentation.pptx",
    "README.md",
    "requirements.txt",
    "app.py",
    "insurance_modeling.py",
    "run_all.py",
]

with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for item in paths_to_pack:
        path = root / item
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and "__pycache__" not in file_path.parts:
                    archive.write(file_path, file_path.relative_to(root))
        elif path.exists():
            archive.write(path, path.relative_to(root))

print(f"PACKED {artifact_zip}")
PY

echo "Done. Download /content/insurance_colab_artifacts.zip from the Colab content view."

DTHS Parkinson's Voice-Feature Project — Easy Setup
==================================================

WHAT THIS PROJECT DOES
- Uploads a CSV containing already-extracted voice features.
- Trains a fuzzy k-NN educational prototype.
- Compares it with tuned KNN and an RBF SVM baseline.
- Shows a demo patient twin dashboard and current-session prediction history.

IMPORTANT
This is NOT a medical diagnostic system. It is for academic demonstration only.
The standard UCI parkinsons.data file contains numerical features, not raw audio.
A live microphone recording cannot be directly entered as these 22 numerical features.
Raw-audio processing would need a separate feature extraction module and validation.

STEP 1 — INSTALL PYTHON
English: Install Python 3.11 or 3.12 from https://www.python.org/downloads/
Tanglish: Python install pannunga. Windows-la install pannumbodhu "Add Python to PATH"
option irundha tick pannunga.

STEP 2 — EXTRACT THIS ZIP
English: Extract the ZIP into a folder, for example Desktop\DTHS_Parkinson_Project.
Tanglish: ZIP file-a right-click panni Extract All select pannunga.

STEP 3 — OPEN TERMINAL
Windows:
- Open the extracted folder.
- Click the folder address bar, type cmd, and press Enter.
Tanglish: Folder open panni address bar-la cmd nu type panni Enter press pannunga.

STEP 4 — CREATE VIRTUAL ENVIRONMENT (recommended)
    py -m venv .venv
    .venv\Scripts\activate

If `py` doesn't work, try `python` instead.
Tanglish: Project-ku thevaiyana Python packages separate-ah install aagum.

STEP 5 — INSTALL LIBRARIES
    python -m pip install --upgrade pip
    pip install -r requirements.txt

STEP 6 — GET DATASET
Search online for "UCI Parkinson's Disease Data Set" and download `parkinsons.data`
from the UCI Machine Learning Repository, if available.
Tanglish: CSV dataset download pannitu, app open aana upload pannunga.
Do not use real patient identifiable data.

STEP 7 — RUN THE WEBSITE
    streamlit run app.py

A local browser page will open. If it does not, copy the local URL shown in the terminal
(usually http://localhost:8501) into your browser.

STEP 8 — RUN THE DEMO
1. Open Train & Evaluate.
2. Upload `parkinsons.data` (or another compatible CSV with a `status` target).
3. Review the metrics table and chart.
4. Open Patient Twin Demo.
5. Edit the numerical feature values if needed and click Run demo prediction.
6. Show the class output and session history during your presentation.

COMMON ERRORS
- "status column not found": CSV must contain a target column named status, or change TARGET_COLUMN in model.py.
- "No usable numerical features": check that the CSV has numeric feature columns.
- "streamlit is not recognized": run `python -m streamlit run app.py`.
- Install issue: confirm the virtual environment is active and rerun `pip install -r requirements.txt`.

PROJECT MODULES FOR REPORT
1. Dataset collection
2. Data preprocessing
3. Fuzzy k-NN classification
4. Baseline comparison and evaluation
5. Digital twin dashboard
6. Session history and care-support concept

RESEARCH PAPER FIDELITY NOTE
The paper describes an optimized fuzzy k-NN that jointly optimizes feature weights and k
using particle swarm optimization (PSO), with cross-validation. This starter project uses
a simpler inverse-distance fuzzy k-NN and tunes standard KNN k using grid search. It is a
working educational baseline, not an exact reproduction of the paper's OF-k-NN algorithm.
To claim exact reproduction, the PSO-based optimization and dataset-specific experiment
protocol must still be implemented and validated.

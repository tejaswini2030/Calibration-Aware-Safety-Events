# Calibration-Aware Safety Events

How camera calibration error affects zone, dwell, and proximity safety alerts, evaluated on WILDTRACK.

Frames from one camera are detected, tracked, and projected onto the ground plane with a homography. Event logic then flags zone entry, dwell, and proximity. The core experiment perturbs the calibration correspondences and measures how that error changes event-level precision, recall, and F1.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the WILDTRACK dataset under `data/` (this checkout has it at `data/Wildtrack_dataset`). Experiment outputs are written to `outputs/`. Both directories are gitignored.

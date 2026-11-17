#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=== 1. Grabbing camera frames into data/ ==="
python3 scripts/save_frames.py

echo "=== 2. Calibrating cam1 extrinsics (ArUco size 0.045m at (0.093, 0.128)) ==="
python3 scripts/calibrate_extrinsics.py --camera cam1 --size 0.045 --x 0.093 --y 0.128 --z 0.0

if [ -f "data/cam2_color.png" ]; then
    echo "=== 3. Calibrating cam2 extrinsics ==="
    python3 scripts/calibrate_extrinsics.py --camera cam2 --size 0.045 --x 0.093 --y 0.128 --z 0.0 || echo "cam2 calibration skipped or failed"
fi

echo "=== Calibration complete! Saved to config/camera_calib.yaml ==="

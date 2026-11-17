# Public demo — checklist

## Day before
- [ ] 5 full autonomous runs logged; `python3 scripts/evaluate_timing.py --out logs/summary.md`
- [ ] Teleop practised (fallback): one full sort under 3 minutes
- [ ] Spare charged laptop, cables, U2D2/OpenCR, power supply, printed ArUco marker, tape measure

## Setup at the venue (15 min)
1. Mount robot (riser if used) and board; tape the reachable ring (`scripts/workspace_check.py`).
2. Cameras on, check topics: `ros2 topic hz /cam1/camera/color/image_raw`
3. Lights changed? Save frames, re-check HSV: `scripts/save_frames.py` + `scripts/tune_hsv.py`
4. Camera moved? Re-calibrate (README §3). Verify one peg at a measured spot.
5. Robot bringup, `h` (home) in teleop, test open/close.
6. `./launch/run_autonomy.sh dry_run` → no warnings.
7. Rotate pegs that get a "predicted yaw misalignment" warning by the printed angle.

## Live run
1. Announce: "Fully autonomous, two RGB-D cameras, no human input." Show the debug image window.
2. `./launch/run_autonomy.sh` — hand near the power switch, teleop terminal open for **SPACE** e-stop.
3. Show the evaluation table printed at the end.
4. If autonomy fails: e-stop, switch to `./launch/run_teleop.sh` and finish by keyboard (teleop points).

## Talking points (30 s)
HSV + morphology for colour · depth → 3-D top face → bird's-eye shape features · ArUco extrinsics ·
analytic IK for OMX · straight-down approach with settle delay · slow stepped insertion ·
vision verification + retry · logged success rate and timing.

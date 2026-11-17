# Beginner's guide — how to finish Trial 1, step by step

This explains **what to do, in what order, and what each piece of code does**, so you
understand it instead of copying it. Do the phases in order; each one ends with a check.

---

## Phase 0 — Understand the goal and the scoring

| Rubric item | Points | What earns it in this repo |
|---|---|---|
| Success (autonomy) | 50 | `run_autonomy.sh` places all 4 pegs by itself |
| Success (teleop) | 30 | `run_teleop.sh` — you drive with the keyboard (do this first: it's your safety net) |
| Colour detection | 20 | `vision/color_segmentation.py` + tuned `config/colors.yaml` |
| Shape detection | 20 | `vision/shape_detection.py` + `vision/scene.py` |
| GitHub README | 10 | `README.md` (already follows the checklist — fill in your repo URL, team names, photos) |
| Fastest execution | 15 | raise `motion.move_speed`, cut delays — **only after** it is reliable |
| Zero failures | 5 | reliable vision + calibration; `evaluate_timing.py` shows "first-try success" |

**Strategy:** get teleop working (30 pts guaranteed) → vision → autonomy → speed.

---

## Phase 1 — Install and run the offline tests (1–2 h, no robot)

Follow README §1. Then:

```bash
python3 -m pytest -q tests
python3 scripts/offline_vision_test.py --synthetic
```
Open `out/cam1_debug.png`: green dots on pegs, pink dots on sockets, with names.

**What just happened?** `peg_trial/utils/synthetic.py` *drew* a fake camera image + depth
image of the table, and the real vision code found every peg and socket in it. This lets you
learn and change the vision code at home without the lab.

✅ Check: all tests pass, errors printed < 3 mm.

---

## Phase 2 — Know your coordinate frames (the #1 source of bugs)

* **Robot base frame** (used everywhere): origin = centre of the OMX base plate,
  **x = forward, y = left, z = up**, metres.
* **Camera optical frame**: x = image right, y = image down, z = straight out of the lens.
* `T_base_cam` (4×4 matrix in `config/camera_calib.yaml`) converts camera → base:
  `p_base = T_base_cam · p_cam`. Its last column is *where the camera is*; the 3×3 block is *how it is rotated*.

Measure and write down: robot base centre on the table, table height relative to the base
(`table_z`; negative if the robot sits on a riser), board thickness.

Run `python3 scripts/workspace_check.py`. It tells you the ring (distance from the base)
where pegs and the board must be. **Mark it on the table with tape.**

---

## Phase 3 — Robot alone: bringup + teleop (lab, ~2 h)

1. Put the arm in the ROBOTIS start pose, power on.
2. Terminal A: `ros2 launch open_manipulator_x_bringup hardware.launch.py`
3. Terminal B: check it's alive:
   ```bash
   ros2 topic echo /joint_states --once
   ros2 control list_controllers        # arm_controller + gripper_controller must be "active"
   ```
4. Test the gripper values (they differ between driver versions!):
   ```bash
   ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.019, max_effort: 5.0}}"
   ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: -0.01, max_effort: 5.0}}"
   ```
   Put the working open/close numbers into `task.yaml → robot.gripper_open / gripper_close`.
5. Terminal B: `ros2 run peg_trial teleop_keyboard`. Press `h` (home), then small `w/a/s/d/r/f` moves.
   Press **SPACE** once on purpose to see the e-stop work, `x` to release.

**How teleop works:** each key changes a target point (x, y, z) of the gripper centre.
`planning/kinematics.py → ik()` turns that point into 4 joint angles, and
`control/arm_interface.py` sends them as a `JointTrajectory` to `/arm_controller/joint_trajectory`.

**How the IK works (kinematics.py):**
1. `q1 = atan2(y, x)` — the base turns to face the target.
2. Now the problem is 2-D (distance `r` and height `z`). Step back from the gripper tip by the
   last link length along the pitch direction (pitch = 90° = pointing down) → wrist point.
3. Shoulder + elbow (links 0.130 m and 0.124 m) reach the wrist point: classic 2-link
   "law of cosines" solution (elbow-up).
4. `q4 = pitch − q2 − q3` keeps the gripper pointing down.
The test `test_ik_fk_roundtrip` proves IK and forward kinematics agree.

✅ Check: you can pick a peg and drop it into a socket with the keyboard. **That's 30 points.**

---

## Phase 4 — Cameras + calibration (lab, ~2 h)

1. `rs-enumerate-devices | grep Serial` → `ros2 launch peg_trial cameras.launch.py serial1:=_XXX serial2:=_YYY`
2. `ros2 topic list | grep cam` — make sure the topic names match `config/camera_calib.yaml`.
3. Mount cameras ~0.5 m above the table, looking (almost) straight down: cam1 over the pegs, cam2 over the board.
   Both should see as much of the table as possible (overlap = robustness).
4. Calibrate both (README §3). **Why ArUco?** The marker has 4 corners at known 3-D spots;
   `solvePnP` finds the camera pose that projects them onto the pixels where they appear.
5. Verification: put one peg at a taped, measured spot; run
   `python3 scripts/save_frames.py && python3 scripts/offline_vision_test.py --data data`
   → the printed x, y must match your measurement within ~5 mm. If not: re-measure the marker
   position, check `--size`, check the marker's top edge points away from the robot.

✅ Check: both cameras report the same peg within ~5 mm of each other and of the tape.

---

## Phase 5 — Colour + shape detection (the 40 vision points)

### Colour (`vision/color_segmentation.py`)
* Image converted BGR → **HSV**. Hue = the colour itself, much more stable than RGB under light changes.
* `cv2.inRange` keeps pixels inside `[lower, upper]` → binary mask. Red needs 2 ranges (hue wraps at 0/180).
* **Morphology:** *opening* removes noise specks, *closing* fills small holes.
* The biggest blob per colour is the peg.

Tune with the real lights ON: `python3 scripts/tune_hsv.py --image data/cam1_color.png`.
Click pixels to see their HSV; move sliders until only the peg is white. Copy into `colors.yaml`.
Leave some margin (± 8 in H, wider in S/V). Also tune `board` so the board is white and the holes black.

### Shape (`vision/scene.py` + `vision/shape_detection.py`)
1. Every peg pixel + its depth → a 3-D point in the base frame (`pixels_to_base`).
2. Keep only the highest 8 mm slab → the **top face** (side walls seen in perspective would distort the shape).
3. Draw those points' true x-y on a 1 mm grid → a perfect bird's-eye outline, even for tilted cameras.
4. Measure 3 numbers: fill of the smallest enclosing **triangle**, fill of the smallest **rectangle**,
   and the **radius ratio** (how round). Compare with the ideal table → nearest = shape.
5. Yaw: corner angles × n, averaged, ÷ n (handles the n-fold symmetry).

Sockets: the board colour is segmented; its outline is filled; *filled − board* = holes. Each hole
is projected onto the board-top plane and classified the same way.

Two cameras: `fuse()` merges by colour (pegs) or by position (sockets), weighting the camera that
looks more straight down. The autonomy node additionally takes the **median of ~6 frames**.

Safety net: if the organisers tell you the colour↔shape pairs, write them in `colors.yaml → shape:`.

Debug live: `ros2 launch peg_trial vision.launch.py` + `ros2 run rqt_image_view rqt_image_view /peg_trial/debug/cam1`,
and `ros2 topic echo /peg_trial/scene`.

✅ Check: 20 different peg placements → all shapes correct. Write the result in the README (judges like numbers).

---

## Phase 6 — Autonomy (50 points)

### Dry run first
```bash
./launch/run_autonomy.sh dry_run      # cameras on, robot NOT moving — prints every planned step
```
Read the output: every peg should get a plan; look for `unreachable` (move it inside the ring) and
`predicted yaw misalignment` warnings.

### How the planner works (`planning/pick_place_planner.py`)
```
home ─► pre-grasp (4 cm above peg, gripper open)
     ─► approach: straight DOWN in 1 cm IK steps (along the board normal)
     ─► settle 0.6 s ─► close ─► wait
     ─► lift straight up until the peg bottom clears the board
     ─► transit (joint space, fast) ─► hover 3 cm above socket
     ─► insert: 5 mm steps, slowly ─► open ─► back up ─► home
```
"Straight line" = many small IK points along the line, sent as one trajectory.

**The yaw problem:** OMX has no wrist-roll joint. When joint 1 turns by Δ while carrying, the
peg turns by Δ too. The planner computes the final misalignment (mod the shape's symmetry —
90° for the square, 120° triangle, 60° hexagon, none for the circle) and warns you.
Fix during setup: rotate that peg by the printed angle. Circle sockets never care.

### Verification + retry (`nodes/autonomy_node.py`)
After each peg the arm goes home (out of the view), takes a fresh median scene and checks:
peg centre within 15 mm of the socket centre **and** peg top ≈ socket top + peg height − insert depth.
Fail → one retry from the newly detected peg position (`execution.max_retries`).

### Then the real thing
Reduce `motion.move_speed` to 0.03 for the first run, hand near the power switch.
`./launch/run_autonomy.sh` → watch → `scripts/evaluate_timing.py`.

Common fixes:
| Symptom | Fix |
|---|---|
| Gripper lands 1 cm off, always in the same direction | calibration offset → redo Phase 4, or check `T_base_cam` translation |
| Peg slips while lifting | larger `grasp_depth` (grip lower), more `close_delay`, more `gripper_max_effort` |
| Peg hits the socket edge | `hover_offset` bigger, `descend_step` smaller, check yaw warning |
| Peg knocked over on approach | increase `pregrasp_offset` (if reachable) |
| "no sockets detected" | tune board HSV; board must be empty at start |
| verify FAIL but peg is in | loosen `verify_xy_tol` / `verify_z_tol`; arm may be blocking a camera |

✅ Check: 5 runs in a row, all pegs placed. Log them (`logs/`) and put the table in the README.

---

## Phase 7 — Speed (bonus) and repo polish

* Speed: `move_speed` ↑, `settle_delay`/`close_delay`/`release_delay` ↓, `descend_time_per_step` ↓,
  `verify: false` only if you trust it. Change ONE thing at a time and re-run the evaluation.
* README: add team names, photos/GIF of the setup, your measured success rate and times, the
  camera mounting photo. Push to GitHub; test the README instructions on a clean machine.
* Demo: follow `docs/demo_guide.md`.

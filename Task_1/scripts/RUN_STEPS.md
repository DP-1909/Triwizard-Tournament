# Trial flow: one peg at a time → all 4 sockets filled

Task: a person puts **one** peg anywhere in the workspace → robot picks it and inserts it in its socket →
next peg is put down → … The pegs already in sockets stay there, so every later move must avoid them.
This is `execution.mode: one_by_one` in `config/task.yaml` (the default).

## A. One-time setup (before the demo day)

1. **Raise the robot** on a firm 8–9 cm block, clamp it. Measure the table top relative to the bottom
   of the robot base → `table_z` (negative, e.g. `-0.085`).
2. **Fix the board** 9–21 cm from the robot base (`python3 scripts/workspace_check.py`). It must not move during the run.
   Socket centres need ≈ 7 cm spacing for your pegs (model is conservative: placed pegs = circles over their corners), else the fingers cannot go down
   next to a peg that is already placed. If possible turn the board so the socket faces point at the robot.
3. **Orientation:** with the board empty and cam1 roughly calibrated, save the scene and run
   `python3 scripts/orientation_check.py --scene scene.json`. It prints how far each peg will arrive
   turned from its socket and where to move the robot base to make that smallest (see §D).
4. **Camera:** fix cam1, then (re)calibrate (README §3) — always after the robot or board moved. Check one peg at a measured spot (±5 mm).
5. **Gripper:** measure the finger gap fully open / fully closed → `robot.gripper_gap_open/closed`.
6. **Colours:** `scripts/tune_hsv.py` under the demo lights; in `colors.yaml` leave `shape: ""` so shape detection is shown.
7. `cd ~/peg_ws && colcon build --symlink-install && source install/setup.bash`, run `python3 -m pytest -q tests`.

## B. Every run

| # | Terminal | Command / action | Check |
|---|---|---|---|
| 1 | A | `ros2 launch open_manipulator_x_bringup hardware.launch.py` | joints stiff, no `Fail` lines |
| 2 | B | `ros2 launch peg_trial cameras.launch.py` | `ros2 topic hz /cam1/camera/color/image_raw` ≈ 30 Hz |
| 3 | — | **Board empty**, no peg on the table | |
| 4 | C | `./launch/run_autonomy.sh dry_run` (robot does not move) | log: `sockets: [4 shapes]`, then `waiting for the next peg` |
| 5 | — | Put a peg down → log shows the plan (grip depth, route) → Ctrl+C | no `no safe plan` |
| 6 | D | `ros2 run peg_trial teleop_keyboard` | keep open: **SPACE = emergency stop** |
| 7 | C | `./launch/run_autonomy.sh` (real run) | robot goes home, finds 4 sockets, says `waiting for the next peg` |
| 8 | — | Put peg 1 anywhere in the ring (not on the board, ≥ 6 cm from sockets), take the hand away | after ~1.5 s still, the robot picks, inserts, goes home, checks with the camera |
| 9 | — | When the log says `waiting for the next peg` again → put peg 2 … repeat until `4/4 pegs placed` | |
| 10 | C | `python3 scripts/evaluate_timing.py --last 1` | success + times for the report |

Rules for the person placing pegs: only when the arm is at **home** and the log says *waiting*;
put the peg **upright**; never reach in while the arm moves.

If the log says **`no safe plan — put the peg somewhere else`** (too far, too close to a placed peg, gripper
too narrow for that rotation): move the peg — usually closer to the robot or further from the board. The robot
notices the new position and continues by itself.

## C. What the robot does for each peg
1. Waits until a new peg (colour not placed yet, not in a socket) stands still.
2. Plans with the collision model: obstacles = pegs already in sockets + board + table.
3. Opens the gripper to the peg's width at its random rotation + 12 mm, goes down vertically, grips 24 mm below the top.
4. Carries it **over** the placed pegs if it can reach that high, else **around** them; inserts slowly.
5. Opens only 8 mm to let go, lifts clear, goes home, verifies with the camera (retries once if it failed).

## D. Orientation (your pegs: square 40 mm, triangle 34 mm sides, hexagon 20 mm sides, circle Ø40 mm)

* The pegs may lie at **any rotation** — that is handled: the gripper opens to the peg's width at that rotation
  (square up to 57 mm across corners → opens ~69 mm) and, when it closes, turns the peg until a flat face lies
  on a finger. After the grasp the peg's rotation relative to the arm is therefore always the same.
* With 4 joints and no wrist roll the arm **cannot turn the peg** after that. The rotation at the socket depends
  only on where the socket is relative to the robot base. Zero error needs:
  square — a side facing the robot base; hexagon — a corner pointing at the base;
  triangle — a side parallel to the line to the base (two mirror cases: the peg lands in the right one only if
  it was lying close to it, ~50 %); circle — any.
* So: (1) move the robot base where `orientation_check.py` says, (2) recalibrate, (3) the remaining error must be
  inside the socket clearance (≈ atan(clearance / half-width): 1 mm on the 40 mm square ≈ 3°).
* The log prints `predicted yaw error` for every peg; `verify` reports a failed insertion and retries once.

# Tri-Wizard Trial 1 — Peg Sorting (OpenMANIPULATOR-X + 2 × RGB-D)

Detect four coloured pegs (square, triangle, circle, hexagon), find the matching
sockets on the board, and pick-and-place each peg — by **keyboard teleop** or **full autonomy**.

```
 2 RGB-D cameras ─► colour segmentation (HSV + morphology)
                  ─► 3-D points in robot frame (depth + calibration)
                  ─► top-face footprint ─► shape (triangle/square/hexagon/circle) + yaw
                  ─► sockets = holes in the board ─► match peg ↔ socket by shape
                  ─► IK planner: pre-grasp → straight-down approach → settle → close → lift
                                 → hover socket → slow insert → release → home
                  ─► verify with vision (in socket?) ─► retry once / log result
```

> New to ROS 2 or this code? Read **[docs/GUIDE.md](docs/GUIDE.md)** first — it explains every part step by step.

---

## 1. Setup

| Item | Version used |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Python | 3.10 (system) |
| Robot | ROBOTIS OpenMANIPULATOR-X (U2D2 or OpenCR) |
| Robot driver | `ROBOTIS-GIT/open_manipulator` branch `humble` (ros2_control) |
| Cameras | 2 × Intel RealSense D435/D435i (`realsense2_camera` ≥ 4.54) — any RGB-D with aligned depth works |
| OpenCV | 4.5.4 from apt (`python3-opencv`, includes `cv2.aruco`) |

```bash
# ROS 2 Humble must already be installed (https://docs.ros.org/en/humble/Installation.html)
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-yaml python3-pytest \
  ros-humble-cv-bridge ros-humble-message-filters ros-humble-control-msgs \
  ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-dynamixel-sdk \
  ros-humble-realsense2-camera ros-humble-rqt-image-view

mkdir -p ~/peg_ws/src && cd ~/peg_ws/src
git clone -b humble https://github.com/ROBOTIS-GIT/open_manipulator.git
git clone <YOUR-REPO-URL> tri-wizard-peg-trial
cd ~/peg_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install          # symlink: edits in config/ apply without rebuild
source install/setup.bash               # do this in EVERY new terminal
```

Check without any hardware (no ROS needed):

```bash
cd ~/peg_ws/src/tri-wizard-peg-trial
python3 -m pytest -q tests                           # 23 tests: IK, vision, planner, calibration
python3 scripts/offline_vision_test.py --synthetic   # fake scene -> out/cam1_debug.png
python3 scripts/workspace_check.py                   # where the robot can reach
```

## 2. Launch files / scripts

| What | Command |
|---|---|
| Robot (real, U2D2) | `ros2 launch open_manipulator_x_bringup hardware.launch.py` |
| Robot (real, OpenCR) | `ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0` |
| Robot (Gazebo) | `ros2 launch open_manipulator_x_bringup gazebo.launch.py` |
| Both cameras | `ros2 launch peg_trial cameras.launch.py serial1:=_<SN1> serial2:=_<SN2>` |
| **Full autonomy** | `./launch/run_autonomy.sh` (= `ros2 launch peg_trial autonomy.launch.py` + evaluation) |
| Autonomy, no robot | `./launch/run_autonomy.sh dry_run` |
| **Teleop** | `./launch/run_teleop.sh` (vision in background + keyboard in this terminal) |
| Vision only | `ros2 launch peg_trial vision.launch.py` then `ros2 run rqt_image_view rqt_image_view /peg_trial/debug/cam1` |

## 3. Camera calibration (both cameras)

* **Intrinsics** (fx, fy, cx, cy) are read live from each camera's `camera_info` — RealSense ships factory-calibrated.
* **Extrinsics** (`T_base_cam`, camera pose in the robot base frame) live in `config/camera_calib.yaml`.
  Robot base frame = OMX `link1`: origin at the centre of the base plate, **x forward, y left, z up**.

```bash
python3 scripts/calibrate_extrinsics.py --make-marker      # print marker_0.png at 100 %, measure the black square
# lay it FLAT on the table, top edge pointing AWAY from the robot, centre at a measured (x, y)
python3 scripts/save_frames.py                              # grabs a frame from each camera into data/
python3 scripts/calibrate_extrinsics.py --camera cam1 --size 0.08 --x 0.20 --y 0.00 --z -0.05
python3 scripts/calibrate_extrinsics.py --camera cam2 --size 0.08 --x 0.20 --y 0.00 --z -0.05
```
`--z` = table height in the base frame (= `table_z` in `config/task.yaml`). Reprojection error should be < 1 px.
Check: put a peg at a measured spot, run `offline_vision_test.py --data data` → printed x, y must match within ~5 mm.

Notes: mount cameras **as top-down as possible (≤ 30° tilt)**, ~0.5 m above the table, one over the pegs and one over the board; enable `align_depth` (done by `cameras.launch.py`); keep the arm's home pose out of the cameras' view; re-calibrate if a camera is bumped.

## 4. Parameter files

| File | Contents |
|---|---|
| `config/colors.yaml` | **HSV thresholds per peg colour** + board colour, morphology kernels, optional fixed colour→shape |
| `config/camera_calib.yaml` | topics + `T_base_cam` for `cam1`, `cam2`, depth scale |
| `config/task.yaml` | table height, peg height, grasp/place offsets, speeds, retries, verification tolerances, robot backend/topics, gripper open/close values |

Tune colours: `python3 scripts/tune_hsv.py --image data/cam1_color.png` (sliders; copy the printed range).

## 5. Usage

### Teleoperation (keyboard)
```
w/s  forward/back   a/d  left/right   r/f  up/down   [ ]  step size
o/c  open/close     g    grasp (down + settle + close)   l  lift 6 cm   p  place (lower slowly + open)
1-4  hover over detected peg (tri, square, hex, circle)   6-9  hover over socket (same order)
h    home           SPACE  EMERGENCY STOP   x  release e-stop   ESC  quit
```
Typical cycle: `1` → `g` → `l` → `6` → `f` (down to just above the socket) → `p`.

### Autonomous
1. Robot bringup + cameras running, board **empty**, pegs standing upright in the reachable ring (`workspace_check.py`).
2. `./launch/run_autonomy.sh` — order of pegs is set by `execution.order` in `task.yaml`.
3. Watch `/peg_trial/status` (`ros2 topic echo /peg_trial/status`) and the debug images.

## 6. Safety and emergency stop

* **SPACE** in the teleop terminal → publishes `true` on `/peg_trial/estop` → every node commands the arm to hold its current position and aborts the motion.
* From any terminal: `ros2 topic pub --once /peg_trial/estop std_msgs/msg/Bool "{data: true}"` (release with `false`).
* `Ctrl+C` on the autonomy terminal also sends a hold command.
* **Hardware stop:** keep a hand on the U2D2 power hub / OpenCR power switch — cutting power drops torque (the arm will sag: keep it over the table).
* Before power-on put the arm in the ROBOTIS recommended start pose; first run every new setup with `dry_run`, then at reduced `motion.move_speed`.
* Keep hands out of the workspace during autonomy; nothing else on the table.

## 7. Evaluation

Every autonomous run appends to `logs/trials.csv` (per peg: success, attempts, time, predicted yaw error) and `logs/runs.csv` (per run: total time).

```bash
python3 scripts/evaluate_timing.py                       # all runs
python3 scripts/evaluate_timing.py --last 5 --out logs/summary.md
```
Reports success rate, first-try success (zero-failure bonus), mean time per peg, best full-run time, per-shape table and failure reasons.

## 8. Known limitations (read before the demo!)

* **No wrist roll on OMX** → the gripper cannot rotate a peg about the vertical. A peg's yaw changes only with joint 1 while it is carried. The planner prints the **predicted misalignment**; rotate that peg by the printed angle during setup, or use sockets with clearance/chamfer. Circles are never affected.
* **Short reach** with a straight-down gripper and 10 cm pegs: max gripper height ≈ 0.13 m above the base. Raise the robot on a ~5 cm riser (`table_z: -0.05`) and keep pegs/board 12–22 cm from the base.
* Gripper opening: check your pegs (4–6 cm) fit between the fingers at `gripper_open`.

## Repository layout
```
tri-wizard-peg-trial/
├─ peg_trial/                 # ROS 2 python package
│  ├─ vision/                 # color_segmentation, shape_detection, pose_estimation, socket_detection, scene
│  ├─ planning/               # kinematics (OMX FK/IK), pick_place_planner
│  ├─ control/                # arm_interface (ros2_control / services / dry_run), teleop_keyboard
│  ├─ nodes/                  # vision_node, autonomy_node
│  └─ utils/                  # config, scene_buffer, calib (ArUco), synthetic (fake RGB-D scenes)
├─ launch/                    # *.launch.py, run_autonomy.sh, run_teleop.sh
├─ config/                    # colors.yaml, camera_calib.yaml, task.yaml
├─ scripts/                   # calibrate_extrinsics, save_frames, tune_hsv, workspace_check,
│                             # offline_vision_test, evaluate_timing
├─ tests/                     # pytest (no ROS needed)
├─ docs/                      # GUIDE.md (beginner walkthrough), demo_guide.md
└─ README.md
```

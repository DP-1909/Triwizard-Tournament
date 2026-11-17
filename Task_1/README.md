# The Tri-wizard Tournament

**Chosen task: Task One — peg sorting with the OpenMANIPULATOR-X**

**Demo video (Task 1):** https://youtu.be/7FnyykQBuIk

This repository holds two tasks:

| Folder | Task | Status |
|---|---|---|
| [`Task_1/`](Task_1/) | **Task One: peg sorting (submitted task)** | Completed and demonstrated on the real robot |
| [`Task_3_Heal/`](Task_3_Heal/) | Task 3: buzz-wire path tracing on the Addverb HEAL arm | Software and lab procedure done, not completed on the robot. See [`Task_3_Heal/README.md`](Task_3_Heal/README.md) |

The rest of this file covers **Task One**.

---

## 1. What we built

A ROS 2 system that lets a **ROBOTIS OpenMANIPULATOR-X** (4-DOF arm with gripper) sort four coloured pegs (**triangle, square, hexagon, circle**) into the matching sockets of a board. An **Intel RealSense RGB-D camera** looks at the table. The robot can be driven in two ways:

- **Full autonomy:** the camera finds the pegs and the sockets, the planner works out a collision-free pick-and-place, and the arm places each peg in its socket.
- **Keyboard teleoperation:** an operator drives the arm with the keyboard, with helper keys that hover over detected pegs and sockets. There is an emergency stop on the space bar.

In the run in the demo, the robot placed **all 4 pegs (4/4) in about 141 s** with no emergency stop. That run is logged in `logs/runs.csv`.

## 2. Our approach

```
 RGB-D camera ─► colour segmentation (HSV + morphology)
              ─► 3-D points in the robot frame (depth + ArUco extrinsic calibration)
              ─► peg top face ─► shape (triangle / square / hexagon / circle) + yaw
              ─► sockets = holes in the board ─► match each peg to its socket by shape
              ─► planner: analytic IK + collision model + A* routing
                 pre-grasp → straight-down approach → settle → close → lift
                 → carry (over / around other pegs) → hover socket → slow insert → release → home
              ─► (optional) check with vision, retry once, log the result
```

1. **Vision.**
   - Each peg colour is segmented in HSV, then cleaned with morphology.
   - Depth turns the pixels into 3-D points. We keep only the top face of each peg and classify its outline as triangle, square, hexagon or circle. We also estimate the peg's rotation (yaw).
   - The sockets are found as holes in the board and are labelled by shape the same way.
2. **Calibration.**
   - Camera intrinsics come from the RealSense `camera_info` topic.
   - The camera's pose in the robot base frame is found with an **ArUco marker** placed at a measured spot on the table (`scripts/calibrate_extrinsics.py`).
3. **Planning.**
   - **Kinematics:** a closed-form (analytic) IK for the 4-DOF OMX arm keeps the gripper pointing straight down.
   - **Collision model:** the model includes the gripper fingers, the slider block and the wrist, plus the pegs already placed, the board and the table. Every straight-line segment is checked every 5–10 mm.
   - **Carrying:** the planner carries a peg **over** the other pegs when the arm can reach that high. Otherwise it routes **around** them with A*.
   - **Gripper opening:** the gripper opens only to the width of the peg at its current rotation, plus a small margin. It grips 18–25 mm below the peg top, so the rest of the peg is free to go into the socket.
4. **Execution.**
   - The `one_by_one` mode is the default: a person puts down one peg, the robot places it, then waits for the next.
   - The robot inserts slowly in small steps, releases with a small opening, then returns home.
   - Each run is written to CSV logs so success rate and timing can be evaluated.
5. **Safety.**
   - The space bar in the teleop terminal publishes on `/peg_trial/estop`, and every node then holds the arm.
   - The `dry_run` backend tests all the logic without moving the robot.
   - Offline tests and synthetic scenes check the code without any hardware.

### Key features
- Fully autonomous pick-and-place driven by vision, plus keyboard teleop as a fallback.
- Shape detection from the 3-D top face, so the same code works for any colour→shape mapping.
- Collision-aware planning that accounts for pegs already in their sockets.
- The gripper opening adapts to each peg's random rotation.
- Emergency stop, `dry_run` mode and offline unit tests.
- Automatic logging and a timing/success report (`scripts/evaluate_timing.py`).
- Helper tools for HSV tuning, camera calibration, workspace reach, orientation and plan checks.

### Technologies used
| Area | Tools |
|---|---|
| Robot | ROBOTIS OpenMANIPULATOR-X, `open_manipulator` (Humble, ros2_control) |
| Middleware | ROS 2 Humble, Ubuntu 22.04, Python 3.10 |
| Camera | Intel RealSense D435 / D435i, `realsense2_camera`, `pyrealsense2` |
| Vision | OpenCV 4.5 (HSV segmentation, contours, `approxPolyDP`, ArUco), NumPy |
| Config / tests | YAML parameter files, `pytest` |

## 3. Repository layout (Task 1)

```
Task_1/
├─ scripts/
│  ├─ peg_ws/                         ★ MAIN RUNNABLE ROS 2 WORKSPACE
│  │  └─ src/
│  │     ├─ tri-wizard-peg-trial/     our ROS 2 package `peg_trial`
│  │     │  ├─ peg_trial/vision/      colour segmentation, shape + socket detection, scene fusion
│  │     │  ├─ peg_trial/planning/    OMX kinematics (IK), collision model, A* path planner, pick-place planner
│  │     │  ├─ peg_trial/control/     arm interface (ros2_control / services / dry_run), keyboard teleop
│  │     │  ├─ peg_trial/nodes/       vision_node, autonomy_node
│  │     │  ├─ peg_trial/utils/       config, ArUco calibration, scene buffer, synthetic scenes
│  │     │  ├─ launch/                cameras / vision / autonomy launch files, run_autonomy.sh, run_teleop.sh
│  │     │  ├─ config/                colors.yaml, camera_calib.yaml, task.yaml
│  │     │  ├─ scripts/               calibration, HSV tuning, workspace / plan / orientation checks, evaluation
│  │     │  ├─ tests/                 pytest (no ROS or robot needed)
│  │     │  └─ docs/                  GUIDE.md, RUN_STEPS.md, COLLISION_AVOIDANCE.md, demo_guide.md
│  │     └─ open_manipulator_driver/  ROBOTIS open_manipulator (humble branch), the robot driver
│  ├─ peg_trial_task1/                copy of the peg_trial package (Task 1 version)
│  ├─ peg_trial_collision/            copy of the package with the collision-avoidance notes
│  ├─ Computer Vison/                 early stand-alone RealSense prototypes (HSV tuner, 3-D peg detector)
│  ├─ socket_centroids/               stand-alone socket locator (ArUco + depth), writes its own README/CSV
│  └─ RUN_STEPS.md                    step-by-step demo procedure
├─ src/, robot_descriptions/, ...     competition starter files (MuJoCo models and examples), not used by the final pipeline
```

Detailed docs for the package are in `Task_1/scripts/peg_ws/src/tri-wizard-peg-trial/`:
- `README.md`: full reference;
- `docs/GUIDE.md`: beginner walkthrough;
- `docs/RUN_STEPS.md`: demo-day steps.

## 4. Setup

**Tested with:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, OpenMANIPULATOR-X on a U2D2 or OpenCR, and an Intel RealSense D435/D435i.

```bash
# 1. ROS 2 Humble must already be installed: https://docs.ros.org/en/humble/Installation.html
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-yaml python3-pytest \
  ros-humble-cv-bridge ros-humble-message-filters ros-humble-control-msgs \
  ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-dynamixel-sdk \
  ros-humble-realsense2-camera ros-humble-rqt-image-view
pip3 install pyrealsense2      # only needed for the stand-alone scripts in "Computer Vison/" and "socket_centroids/"

# 2. Get the code
git clone https://github.com/DP-1909/Triwizard-Tournament.git
cd Triwizard-Tournament/Task_1/scripts/peg_ws

# 3. Robot driver: if src/open_manipulator_driver is empty after cloning, fetch it:
#    rm -rf src/open_manipulator_driver
#    git clone -b humble https://github.com/ROBOTIS-GIT/open_manipulator.git src/open_manipulator_driver

# 4. Build
rm -rf build install log        # clean build on a new machine
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash       # do this in EVERY new terminal
```

**Quick check without any hardware** (no ROS needed):
```bash
cd ~/…/Triwizard-Tournament/Task_1/scripts/peg_ws/src/tri-wizard-peg-trial
python3 -m pytest -q tests                           # IK, vision, planner, collision, calibration tests
python3 scripts/offline_vision_test.py --synthetic   # fake scene -> out/cam1_debug.png
python3 scripts/workspace_check.py                   # where the robot can reach
```

## 5. How to run

Open a separate terminal for each step and source the workspace first (`source install/setup.bash` in `peg_ws`). All the `./launch/...` and `scripts/...` commands are run from `peg_ws/src/tri-wizard-peg-trial`.

| # | What | Command |
|---|---|---|
| 1 | Robot bringup (real, U2D2) | `ros2 launch open_manipulator_x_bringup hardware.launch.py` |
|   | Robot bringup (OpenCR) | `ros2 launch open_manipulator_x_bringup hardware.launch.py port_name:=/dev/ttyACM0` |
|   | Robot in Gazebo instead | `ros2 launch open_manipulator_x_bringup gazebo.launch.py` |
| 2 | Camera | `ros2 launch peg_trial cameras.launch.py serial1:=_<SERIAL>` (add `single:=false serial2:=_<SERIAL2>` for a second camera) |
| 3 | Test the logic, robot does not move | `./launch/run_autonomy.sh dry_run` |
| 4 | Keep a teleop terminal open for the e-stop | `ros2 run peg_trial teleop_keyboard` (**SPACE = emergency stop**) |
| 5 | **Full autonomous run** | `./launch/run_autonomy.sh` |
| 6 | Results of the last run | `python3 scripts/evaluate_timing.py --last 1` |
| – | **Teleoperation mode** | `./launch/run_teleop.sh` (vision in the background, keyboard in this terminal) |
| – | Vision only (debug view) | `ros2 launch peg_trial vision.launch.py`, then `ros2 run rqt_image_view rqt_image_view /peg_trial/debug/cam1` |

**Autonomous run, step by step:**
1. Start with the board **empty** and wait until the log says `waiting for the next peg`.
2. Put **one** peg upright in the reachable ring, then take your hand away.
3. After about 1.5 s with the peg still, the robot picks it, inserts it and goes home.
4. Put the next peg down when the log says `waiting for the next peg` again. Repeat until you see `4/4 pegs placed`.

**Teleop keys:**
```
w/s  forward/back   a/d  left/right   r/f  up/down   [ ]  step size
o/c  open/close     g  grasp   l  lift 6 cm   p  place (lower slowly + open)
1-4  hover over detected peg (tri, square, hex, circle)   6-9  hover over socket (same order)
h    home   SPACE  EMERGENCY STOP   x  release e-stop   ESC  quit
```

### Calibration and tuning (when the setup changes)
- **Camera extrinsics** (ArUco marker at a measured spot): run `scripts/do_calibration.sh`, or by hand:
  ```bash
  python3 scripts/calibrate_extrinsics.py --make-marker      # print the marker
  python3 scripts/save_frames.py
  python3 scripts/calibrate_extrinsics.py --camera cam1 --size <marker_m> --x <m> --y <m> --z <table_z>
  ```
- **Colours:** `python3 scripts/tune_hsv.py --image data/cam1_color.png`, then copy the ranges into `config/colors.yaml`.
- **Task parameters:** `config/task.yaml` holds table height, peg sizes, grasp and place heights, speeds, execution mode, peg order and robot backend.
- **Plan check before a run:**
  ```bash
  ros2 topic echo --once --full-length /peg_trial/scene --field data > scene.json
  python3 scripts/plan_check.py --scene scene.json
  ```

### Stand-alone vision scripts (optional, RealSense + `pyrealsense2`)
```bash
cd Task_1/scripts/"Computer Vison"
python3 hsv_tuner.py                   # live HSV sliders, click a pixel to read its HSV
python3 peg_detector_3D.py             # live colour + shape + 3-D position of pegs
python3 peg_detecter_robot_frame.py    # pegs in the robot frame via ArUco marker

cd ../socket_centroids
python3 locate_sockets.py              # socket centres -> README.md + sockets.png (add --live to keep running)
```

## 6. Safety
- **SPACE** in the teleop terminal stops the arm. From any terminal you can also run `ros2 topic pub --once /peg_trial/estop std_msgs/msg/Bool "{data: true}"`.
- `Ctrl+C` on the autonomy terminal also sends a hold command.
- Keep a hand near the U2D2/OpenCR power switch, and run every new setup with `dry_run` first.

## 7. Known limitations
- The OMX has **no wrist roll**, so it cannot rotate a peg after grasping it. The planner prints the predicted misalignment at the socket. `scripts/orientation_check.py` suggests where to put the robot base to keep that error small.
- The reach is short with a straight-down gripper and 10 cm pegs. Raising the robot on a riser (`table_z` in `task.yaml`) helps.

## 8. Demo video

▶ **https://youtu.be/7FnyykQBuIk**

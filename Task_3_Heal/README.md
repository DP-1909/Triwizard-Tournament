# Task 3 (HEAL): buzz-wire path tracing with the Addverb HEAL manipulator

**Demo video:** none. The task was **not completed** on the real robot, so there is no demo video (see [Status](#5-status-and-what-is-left)).

---

## 1. What we did

The task was to make the **Addverb HEAL 6-DOF arm** trace a given path. The path is a **buzz-wire game**: a bent aluminium wire mounted on a wooden board. HEAL has to:
1. go to the left end of the wire;
2. close its gripper on the handle of a hoop that rests there;
3. move the hoop's loop along the whole wire to the right end **without touching the wire**.

HEAL is controlled in **velocity mode** through ROS 2. All our programs publish six joint velocities (rad/s) as `std_msgs/Float64MultiArray` on `/velocity_controller/commands`. They read joint positions from `/joint_states`.

We wrote software for **two approaches**:
1. Teach-and-replay. This could not be used because of how free drive behaved in the lab.
2. A hard-coded buzz-wire path with calibration and a touch-up step.

We also measured the wire and hoop geometry, wrote a full lab procedure, and tested every code path in simulation.

### Files

| File | Purpose |
|---|---|
| `heal_joint_replay.py` | Teach-and-replay in **joint space**: records `/joint_states`, smooths the data, replays it with feedforward + feedback, and analyses the tracking error. Also holds the shared control loop and URDF kinematics. |
| `heal_ee_replay.py` | Teach-and-replay when only **end-effector poses** are available: records any pose topic, converts it offline to joint angles (IK), then replays through the joint script. |
| `heal_buzz_wire.py` | **Hard-coded buzz-wire solution**: jog/teach tool, motion planner, touch-up of the path, and execution. |
| `BUZZ_WIRE.md` | Detailed lab procedure for the buzz-wire solution. |
| `RUNBOOK.md` | Lab procedure for the original teach-and-replay plan. |

The scripts do **not** import the course starter code. The parts we needed (quintic trajectories and kinematics) are re-implemented with NumPy inside the scripts. Generated data (CSV files, plots, marks, plans, logs) goes to `Task_3_Heal/data/`.

## 2. Approach and methodology

### 2.1 Understanding the starter code
- We read the provided starter code:
  - the robot start-up README;
  - `quintic_planner.py`;
  - `dls_velocity_commander.py`, the Cartesian damped-least-squares controller;
  - `heal_end_effector_pub_node.py`;
  - the homing scripts;
  - the gripper wrappers.
- Every script there controls HEAL by publishing joint velocities. The velocity controller only runs while `heal_server` and `velocity_control.launch.py` are running.
- We found a few issues in the starter code:
  - a hard-coded workspace path in the gripper wrapper;
  - a missing topic name in the optional end-effector subscription;
  - no joint-limit or collision checks.

### 2.2 First approach: teach by demonstration (record in free drive, replay)
- **Plan:** guide the arm by hand in `./free_drive` while recording `/joint_states`, then switch to `heal_server` and replay the motion.
- **`heal_joint_replay.py`** has these sub-commands:
  - `check`, `record`, `process` (trim, resample at 100 Hz, smooth, differentiate);
  - `replay` and `analyze` (joint and end-effector tracking error, with plots);
  - plus `save-urdf`, `goto`, `teach-waypoints`, `from-waypoints` and `demo-data`.
- **Replay control law** (100 Hz):

  `q̇_cmd = s·q̇_ref(s·t) + Kp·(q_ref(s·t) − q)`, where `s` is the speed scale.

  The command is clamped in velocity and acceleration. Motion stops on a large tracking error, on stale joint states, or on Ctrl+C.
- **`heal_ee_replay.py`** covers the case where only end-effector poses are available. It records any pose-like topic and solves damped-least-squares IK offline, `dq = Jᵀ(JJᵀ + λ²I)⁻¹e`, warm-starting each sample from the previous one. It then replays through the joint script.
- We tested both pipelines in simulation and against a fake ROS 2 interface.
- **In the lab:** free drive published **neither joint states nor end-effector poses**, so this approach could not be used.

### 2.3 Second approach: hard-coded buzz-wire path (`heal_buzz_wire.py`)
1. **Measurements.**
   - We read 18 centre-line corner points of the wire from annotated photos, scaled with measured distances. Accuracy is about ±3 mm.
   - The hoop's loop is a small triangle with sides 3.5 / 2.5 / 2.5 cm, which leaves only **about 3.5 mm of free space** around the wire.
2. **Calibration.** We teach three poses with an interactive Cartesian jog tool:
   - `left` and `right`: the gripper centre on the two wire ends. These fix where the wire is in HEAL's base frame.
   - `grasp`: the fingers around the hoop handle. This fixes where the loop is relative to the gripper.
3. **Motion planning:**
   - **Approach:** straight-line pre-grasp → grasp, then close the gripper.
   - **Flip:** rotate the hoop about its loop centre so the handle points at HEAL, out of the wire's plane, so the gripper never sweeps through the neighbouring bends.
   - **Trace:** move the loop centre along the rounded wire centre-line, turning it at each bend so it stays perpendicular to the wire.
   - IK turns every pose into a joint trajectory. The timing slows down in the corners and wherever a joint would exceed 0.25 rad/s.
4. **Safety checks in the planner:**
   - joint limits;
   - reachability;
   - whether the gripper is above the handle;
   - gripper clearance above the board and mat, and from the wire plane.

   If any check fails, `run` asks for an explicit `yes` before it moves.
5. **Touch-up.** Photo measurements can't reach 3.5 mm accuracy. So the robot stops at every bend, and the operator nudges the loop to the centre in 1 mm steps. The corrections are saved and used by the next `plan`.
6. **Testing.** Simulated with a 6-DOF model and a simulated ROS 2 interface. The loop stayed within about **0.9 mm** of the planned path.

   The simulation also flagged a real risk: at the bottom of the vertical stretches, the gripper body comes within about 0–2 cm of the board. Gripping near the end of the handle helps most.

## 3. Setup

**Requirements:**
- Python 3 with **NumPy**. **matplotlib** is optional and only used for plots.
- For the real robot: **ROS 2 Humble** with `rclpy`, and the Addverb HEAL workspace (`addverb_cobot_control`, `addverb_cobot_msgs`) built and sourced.
- Offline rehearsal (`--sim`, `process`, `analyze`, `demo-data`) needs **no ROS**.

```bash
pip3 install numpy matplotlib
git clone https://github.com/DP-1909/Triwizard-Tournament.git
cd Triwizard-Tournament/Task_3_Heal
```

**Robot bring-up** (lab PC). These are the paths and IP from our lab, so adjust them to yours:
```bash
# T1: start the robot
ssh cobot@192.168.1.25
cd /home/cobot/addverb_images/gripper_fix_2026-09-08/
sudo env HOME=/home/cobot ./cobot run        # type 1 at the prompt
./heal_server

# T2: velocity controller (only while heal_server runs)
cd ~/<addverb_ws> && source install/setup.bash
ros2 launch addverb_cobot_control velocity_control.launch.py

# T3: our scripts
cd ~/<addverb_ws> && source install/setup.bash
cd <path>/Triwizard-Tournament/Task_3_Heal
```

## 4. How to run

Every command has `-h`, for example `python3 heal_buzz_wire.py run -h`.

### 4.1 Rehearse without the robot (no ROS)
```bash
python3 heal_joint_replay.py demo-data --name demo          # fake recording
python3 heal_joint_replay.py process   --name demo          # -> data/demo_traj.csv + plot
python3 heal_joint_replay.py replay    --name demo --sim    # simulated replay
python3 heal_joint_replay.py analyze   --name demo --sim    # error report + plots
```

### 4.2 Buzz-wire solution on HEAL (with `heal_server` + `velocity_control.launch.py` running)
```bash
python3 heal_joint_replay.py save-urdf                     # save the robot model to data/heal.urdf
python3 heal_joint_replay.py goto --delta 0,0,0,0,0,0.1    # wrist test move: only joint 6 should turn
python3 heal_joint_replay.py goto --delta 0,0,0,0,0,-0.1   # ... and back
python3 heal_buzz_wire.py where                            # tool axes + gripper centre
python3 heal_buzz_wire.py jog                              # teach: mark left, mark right, mark grasp
python3 heal_buzz_wire.py plan                             # build + check the motion -> data/buzz_plan.png
python3 heal_buzz_wire.py run --sim                        # rehearse in simulation
python3 heal_buzz_wire.py run                              # dry run with the hoop REMOVED
python3 heal_buzz_wire.py touchup                          # centre the loop at every bend (1 mm nudges)
python3 heal_buzz_wire.py plan                             # re-plan with the saved corrections
python3 heal_buzz_wire.py run                              # final run
```

**Useful options:**
- `--speed 0.7`: slower.
- `--auto`: no ENTER between phases.
- `--release`: open the gripper at the end.
- `--no-gripper`: you operate the gripper yourself.
- `--fit-span`: stretch the wire to the taught left–right span.

Before the first real run, set the board, mat and gripper sizes in the `EDIT HERE` block of `heal_buzz_wire.py`.

### 4.3 Teach-and-replay (only if free drive publishes joint states or poses)
```bash
python3 heal_joint_replay.py check --seconds 8            # does /joint_states change in free drive?
python3 heal_joint_replay.py record  --name path1         # in free drive
python3 heal_joint_replay.py process --name path1
python3 heal_joint_replay.py replay  --name path1 --speed 0.3   # back in heal_server mode
python3 heal_joint_replay.py analyze --name path1
# End-effector-only variant: heal_ee_replay.py check / record / process / ik / replay / analyze
```

Full step-by-step procedures, with what to check at each step:
- [`BUZZ_WIRE.md`](BUZZ_WIRE.md) for the buzz-wire solution;
- [`RUNBOOK.md`](RUNBOOK.md) for teach-and-replay.

**Safety:** keep a hand on the e-stop during every motion. The scripts send zero velocity on large tracking error, on stale joint states, or on Ctrl+C.

## 5. Status and what is left

**The task was not completed on the robot. None of the motion code has been validated on the real HEAL.**

Why we could not finish:
- **Limited lab time.** We had a single 2-hour slot. Start-up and switching between robot modes (`heal_server`, `free_drive`) took a large part of it.
- **No prior experience with HEAL.** We had not used its vendor software, ROS 2 interfaces or gripper service before. We learned them from the starter code during the project.
- **Free drive gave no data,** so the teach-and-replay plan failed in the lab. We had to switch to a hard-coded path late in the project.
- **Very tight accuracy.** There is only about 3.5 mm of free space around a hand-bent wire, and the wire's lowest points are about 1.7 cm above the board. This needs on-robot calibration and touch-up time that we did not have.

What would be needed to finish:
1. One more lab session of about 1.5–2 h to:
   - enter the measured sizes;
   - check the tool axes;
   - teach the three poses;
   - do the dry run and the touch-up.
2. Confirm the gripper service. We assumed `/gripper_controller/command` of type `addverb_cobot_msgs/srv/Gripper`, with 1.0 = open and 0.0 = close.
3. Confirm the clearance at the bottom of the vertical stretches. If it's too tight, grip the handle further out or raise the board.

## 6. Demo video

Not available: the task was not completed on the robot.

## 7. Acknowledgement

The code and documentation were developed with the help of an AI assistant (Claude, Anthropic). Measurements, lab work and decisions were done by the group.

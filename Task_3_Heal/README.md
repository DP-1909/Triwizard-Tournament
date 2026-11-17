# Project 3, Group 3: Path tracing with the Addverb HEAL manipulator

## 1. Task

The task was to make the HEAL 6-DOF manipulator trace a given path. The path turned out to be a **buzz-wire game**: a bent aluminium wire mounted on a wooden board. HEAL has to:
1. go to the left end of the wire;
2. close its gripper on the handle of a hoop that rests there;
3. move the hoop's loop along the whole wire to the right end **without touching the wire**.

HEAL is controlled in **velocity mode** through ROS 2. Every program ends up publishing six joint velocities (rad/s) as `std_msgs/Float64MultiArray` on `/velocity_controller/commands`. Joint positions are read from `/joint_states`.

## 2. Files in this folder

| File | Purpose |
|---|---|
| `heal_joint_replay.py` | Teach-and-replay in **joint space**. Records `/joint_states`, smooths the data, replays it with feedforward + feedback, and analyses the tracking error. Also contains the shared control loop and the URDF kinematics. |
| `heal_ee_replay.py` | Teach-and-replay when only **end-effector poses** are available. Records any pose topic, converts it offline to joint angles (inverse kinematics), then replays with the joint script. |
| `heal_buzz_wire.py` | **Hard-coded buzz-wire solution**: jog/teach tool, motion planner, touch-up of the path, and execution. |
| `BUZZ_WIRE.md` | Detailed lab procedure for the buzz-wire solution. |
| `RUNBOOK.md` | Lab procedure for the original teach-and-replay plan. |
| `README.md` | This file. |

All scripts sit next to `Project3 Instructions/` (the starter code), but they **do not import it**. The parts we needed (quintic trajectories and kinematics) are re-implemented inside the scripts with numpy, so missing PyKDL or a wrong folder path cannot break them. Generated data goes to `Group_3/data/`.

## 3. Steps we followed

### 3.1 Understanding the starter code
1. We read all files in `Project3 Instructions/`:
   - the README for starting the robot;
   - `quintic_planner.py` (quintic joint trajectories);
   - `dls_velocity_commander.py` (Cartesian damped-least-squares controller);
   - `heal_end_effector_pub_node.py` (end-effector pose from forward kinematics);
   - the homing scripts and the gripper wrappers.
2. We noted that every script controls HEAL by publishing joint velocities, and that the velocity controller only runs while `heal_server` and `velocity_control.launch.py` are running.
3. We noted some issues in the starter code:
   - a hard-coded workspace path in the gripper wrapper;
   - a missing topic name in the optional end-effector subscription of the DLS commander;
   - no joint-limit or collision checks in any script.

### 3.2 First approach: teach by demonstration (record in free drive, replay)
4. **Plan:** guide the arm by hand in `./free_drive` while recording `/joint_states` with timestamps, then switch to `heal_server` and replay the motion.
5. We wrote `heal_joint_replay.py` with these commands:
   - `check`: inspect the joint-state topic (rate, names, whether values change);
   - `record`: record joint states;
   - `process`: trim the still parts, resample at 100 Hz, smooth, and compute velocities;
   - `replay`: move smoothly to the start, then trace;
   - `analyze`: compute joint and end-effector tracking error, with plots.

   The replay control law is

   `q̇_cmd = s·q̇_ref(s·t) + Kp·(q_ref(s·t) − q)`

   where `s` is the speed scale. The result is clamped in velocity and acceleration. There are safety stops on a large tracking error, on stale joint states and on Ctrl+C.
6. We wrote `heal_ee_replay.py` for the case where only end-effector poses are available. It records any pose-like topic and solves damped-least-squares inverse kinematics offline (`dq = Jᵀ(JJᵀ + λ²I)⁻¹e`), warm-started from the previous sample. It then replays through the joint script.
7. We tested both pipelines in simulation and with a fake ROS 2 interface, including recording, replay, Ctrl+C and the safety aborts.
8. **In the lab:** we started HEAL as described in its README and checked the topics in free drive. **Free drive published neither joint states nor an end-effector pose**, so this approach could not be used.

### 3.3 Second approach: hard-coded buzz-wire path
9. **Measurements:** from annotated photos of the setup we read 18 centre-line corner points of the wire. The scale came from the measured 30 cm span and the 30 cm height of the right end above the table. We also used the measured 3 cm bend spacing and the 10.5 cm bend height. Estimated accuracy is about ±3 mm.

   We measured the hoop:
   - triangular loop with sides 3.5 / 2.5 / 2.5 cm;
   - 9.7 cm handle;
   - largest circle inside the loop: radius about 7 mm, which leaves about 3.5 mm of free space around the wire.
10. **Calibration design:** the position of the board relative to HEAL is unknown, so three poses are taught with an interactive Cartesian jog tool in `heal_server` mode:
    - `left`: gripper centre on the wire's left end;
    - `right`: gripper centre on the wire's right end;
    - `grasp`: open fingers around the hoop handle.

    `left` and `right` fix the wire's position and direction in the robot's base frame. `grasp` fixes where the loop centre is relative to the gripper.
11. **Motion design** (`heal_buzz_wire.py plan`):
    - **Approach:** a straight-line move from the pre-grasp pose to the grasp pose, then close the gripper.
    - **Flip:** rotate the hoop about its loop centre so the handle points at HEAL, perpendicular to the wire's plane, and the loop is exactly square to the wire. With the handle out of the plane, the gripper never sweeps through the neighbouring bends.
    - **Trace:** move the loop centre along the rounded wire centre-line. At each point the loop rotates about the handle axis by the change in wire direction, so it stays perpendicular to the wire.
    - All motions are converted to joint trajectories with inverse kinematics.
    - The timing slows down in the corners. It also slows down locally wherever a joint would exceed 0.25 rad/s.
12. **Safety checks in the planner:**
    - joint limits;
    - reachability;
    - whether the gripper sits above the handle;
    - approximate clearance of the gripper body above the board and mat, and from the wire plane.

    `run` asks for an explicit `yes` if any check fails.
13. **Touch-up** (`heal_buzz_wire.py touchup`): photo measurements cannot reach 3.5 mm accuracy, so the robot moves the hoop to each bend and stops. The operator nudges the loop to the centre in 1 mm steps. The corrections are saved and used in the next plan.
14. **Testing:** we tested planning, flip, trace, touch-up and execution with a simulated 6-DOF arm and a simulated ROS 2 interface. In simulation the loop stayed within about 0.9 mm of the planned path.

    The simulation also showed a real risk: at the bottom of the vertical stretches the gripper body comes within about 0–2 cm of the board. It depends on where the handle is gripped. Gripping near the end of the handle helps most.
15. We prepared a lab procedure (`BUZZ_WIRE.md`): measure, start up, check axes, teach the poses, plan, simulate, dry run without the hoop, touch-up with the hoop, final run.

## 4. How to run (summary)

In a terminal with the workspace sourced, while `heal_server` and `velocity_control.launch.py` are running, from inside `Group_3`:

```bash
python3 heal_joint_replay.py save-urdf                 # save the robot model
python3 heal_joint_replay.py goto --delta 0,0,0,0,0,0.1  # wrist test move (then -0.1 back)
python3 heal_buzz_wire.py where                        # tool axes, gripper centre
python3 heal_buzz_wire.py jog                          # teach: mark left, mark right, mark grasp
python3 heal_buzz_wire.py plan                         # build and check the motion
python3 heal_buzz_wire.py run --sim                    # rehearse in simulation
python3 heal_buzz_wire.py run                          # dry run with the hoop removed
python3 heal_buzz_wire.py touchup                      # centre the loop at every bend
python3 heal_buzz_wire.py plan                         # re-plan with the corrections
python3 heal_buzz_wire.py run                          # final run
```

See `BUZZ_WIRE.md` for the full procedure, including what to check at each step.

## 5. What would be needed to finish

1. **One more lab session (about 1.5–2 h)** to:
   - enter the measured board, mat and gripper sizes;
   - check the tool axes;
   - teach the three poses;
   - do the dry run and the touch-up.
2. **Check the gripper interface on the robot.** We assumed the service `/gripper_controller/command` of type `addverb_cobot_msgs/srv/Gripper`, with position 1.0 = open and 0.0 = close, taken from the starter wrapper.
3. **Confirm the clearance at the bottom of the vertical stretches.** If needed, grip the handle further out or raise the board slightly.
4. **Earlier familiarisation with HEAL.** Knowing in advance what each robot mode publishes would have avoided spending the lab slot on the free-drive approach.

## 6. Acknowledgement

The code and documentation were developed with the help of an AI assistant (Claude, Anthropic). Measurements, lab work and decisions were done by the group.

## 7. Conclusion: the task was not completed

**We could not complete the task on the robot.** The main reasons:

- **Limited lab time.** We had a single 2-hour slot on HEAL. Starting the robot, switching between its modes (`heal_server`, `free_drive`) and checking what each mode publishes took a large part of it. After we learned that free drive was not usable for recording, no time was left to measure, calibrate and test a new approach on the robot.
- **Limited prior knowledge of HEAL.** The group had no earlier experience with HEAL, its vendor software (`./cobot run`, `heal_server`, `free_drive`, `base_rigid`), its ROS 2 interfaces or its gripper service. We learned these from the starter code during the project. That made progress on the real robot slow and made us cautious about running untested motions.
- **Free drive gave no data.** Our first plan was to guide the arm by hand in free drive, record the motion and replay it. In the lab, free drive published **neither joint states nor end-effector poses**, so nothing could be recorded. We had to switch to a hard-coded path late in the project.
- **The task needs very high accuracy.** The hoop's loop is a small triangle (sides 3.5, 2.5 and 2.5 cm). That leaves only about **3.5 mm of free space** around the wire. The wire is hand-bent, and its lowest points are only about 1.7 cm above the board. A path built from ruler or photo measurements is not accurate enough on its own. The calibration and touch-up steps needed for this precision take robot time that we did not have.

**What we did complete:**
- Software for two approaches: teach-and-replay, and a hard-coded buzz-wire path with calibration and touch-up.
- Measurement of the wire and hoop geometry.
- A full step-by-step lab procedure.
- Testing of all code paths in simulation, including a simulated ROS 2 interface.

**None of the motion code has been validated on the real HEAL.**

> *(Group: edit this list to state exactly which steps you managed on the robot, e.g. robot start-up, the topic checks in free drive, `save-urdf`, the wrist test move, teaching the poses.)*

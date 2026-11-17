# Group 3 – HEAL path tracing: lab runbook (2-hour slot)

Two scripts, both in this folder (`Group_3/`, next to `Project3 Instructions/`):

| Script | Use it when | What it does |
|---|---|---|
| `heal_joint_replay.py` | `/joint_states` **changes** while you move the arm in `./free_drive` | record joints → smooth → replay in joint space with feedback |
| `heal_ee_replay.py` | free drive only gives an **end-effector position/pose** | record pose → smooth → offline IK → replay with the joint script |

Both scripts are self-contained (numpy only; matplotlib for plots). They do **not** import the starter code, so the folder location and the starter function signatures cannot break them. They use the same topics as the starter code: `/joint_states` (in) and `/velocity_controller/commands` (out, `Float64MultiArray`, 6 values in rad/s, order joint1…joint6).

Everything lands in `Group_3/data/` (CSV files, PNG plots, text reports).

Every command has `-h`, e.g. `python3 heal_joint_replay.py replay -h`.

---

## 0. Tonight (15 min, on your laptop, no robot)

Rehearse the whole pipeline on a simulated arm so nothing is new tomorrow (on Windows type `python` instead of `python3`):

```bash
cd Group_3
python3 heal_joint_replay.py demo-data --name demo          # fake recording
python3 heal_joint_replay.py process   --name demo          # -> data/demo_traj.csv + demo_traj.png
python3 heal_joint_replay.py replay    --name demo --sim    # simulated replay
python3 heal_joint_replay.py analyze   --name demo --sim    # error report + plots
```

Also, before the slot:
- Copy `Group_3/` to a pen drive (and/or push it to git) so you can put it on the lab PC next to `Project3 Instructions/`.
- Agree on roles: **operator** (keyboard), **e-stop holder** (hand on the e-stop for every motion, no exceptions), **arm guide** (moves the arm in free drive), and someone filming the final run.
- Decide the exact path, and practise tracing it by hand slowly and smoothly.

---

## 1. Time plan for the 2 hours

| Time | Do |
|---|---|
| 0:00–0:15 | Bring-up in `heal_server` mode: launch, `check`, `save-urdf`, one tiny test move (§2) |
| 0:15–0:30 | Switch to `free_drive`, run the checks, **decide Branch A / B / C** (§3) |
| 0:30–0:45 | Record 2–3 takes of the path, process them, look at the plots |
| 0:45–1:20 | Back to `heal_server`: replay at 0.3× → analyze → 0.5× → … |
| 1:20–1:40 | Final runs at the best speed, film them, save reports |
| 1:40–2:00 | Buffer. Copy `data/` off the PC, shut down cleanly (§7) |

If you're behind at 0:45, skip extra takes and replay the best one you have.

---

## 2. Bring-up (heal_server mode) – ~15 min

Open **4 terminals** on the workstation:
- **T1**: robot (ssh)
- **T2**: launch file
- **T3**: your scripts
- **T4**: spare, for `ros2 topic echo` and similar

In **T2, T3, T4**, source the workspace first (the README path; adjust if your PC differs):
```bash
cd ~/Debojit_WS/addverb_ws/ && source install/setup.bash
```
(If `ros2` is not found, run `source /opt/ros/humble/setup.bash` first.)

**Step 1. Start the robot (T1)**
```bash
ssh cobot@192.168.1.25
cd /home/cobot/addverb_images/gripper_fix_2026-09-08/
sudo env HOME=/home/cobot ./cobot run        # type 1 at the prompt
./heal_server
```

**Step 2. Start the velocity controller (T2), only while heal_server is running**
```bash
ros2 launch addverb_cobot_control velocity_control.launch.py
```
Leave it running and wait until it reports the controller loaded/active.

**Step 3. Go to your folder (T3)**
```bash
cd <path>/Group_3
```

**Step 4. Check joint states (T3)**
```bash
python3 heal_joint_replay.py check --seconds 5
```
Expect `/joint_states` at roughly 100+ Hz, with names that include joint1…joint6.
If the names differ, add `--joints nameA,nameB,...` (in controller order) to **every** command.

**Step 5. Save the robot model (T3).** This is needed for the end-effector plots and for IK.
```bash
python3 heal_joint_replay.py save-urdf
```
It prints the chain `base_link -> end-effector` and the joint limits.
If it can't find the tip link, it lists all links. Pick the right one and add `--tip-link NAME` to every command from then on.

**Step 6. First motion test.** This checks that the command order matches the joints. E-stop holder ready.
```bash
python3 heal_joint_replay.py goto --delta 0,0,0,0,0,0.1     # type y
```
Only the **last joint (wrist, joint 6)** should rotate by about 6°. Then send it back:
```bash
python3 heal_joint_replay.py goto --delta 0,0,0,0,0,-0.1
```
- **A different joint moved:** STOP. The controller's joint order isn't joint1…6. Ask the TA for the order (or check `ros2 control list_controllers -v` / the controller yaml) and pass it with `--joints`.
- **Nothing moved and it warned "nobody subscribes":** the velocity controller isn't active. Check T2 and run `ros2 control list_controllers`.

---

## 3. Free-drive check and the decision – ~15 min

**Step 7. Switch to free drive.**
1. Stop the launch in T2 (Ctrl+C).
2. Stop `heal_server` in T1 (Ctrl+C).
3. Run `./free_drive` in T1.

> **Ask the TA once:** does the arm hold its pose when heal_server or free_drive stops? Have the arm guide support the arm while modes switch until you know.

**Step 8. While the arm guide moves the arm, run (T3):**
```bash
python3 heal_joint_replay.py check --seconds 8
python3 heal_ee_replay.py   check --seconds 8
```
Also useful in T4: `ros2 topic list`, then `ros2 topic echo /joint_states --once`.

**Decide:**

| What you see | Branch |
|---|---|
| `check` says "Values change -> joint recording will work" | **A** (§4). The best case. |
| No usable joint states, but `heal_ee_replay.py check` shows a pose-like topic whose values change | **B** (§5). Note the topic name, whether it says *probably MILLIMETRES*, and whether orientation is YES or NO. |
| Nothing changes / no topics | First ask the TA whether a launch file is needed during free drive (try it only if they agree). Otherwise **C** (§6). |

---

## 4. Branch A: joint recording (main plan)

**A1. Record (free drive, T3).** Do 2–3 takes: `path1`, `path2`, …
```bash
python3 heal_joint_replay.py record --name path1
```
- Put the arm at the start of the path, then press **ENTER**.
- Hold still for ~1 s, then trace the path **slowly and steadily**. Slow is fine, because replay speed is set later.
- Hold still for ~1 s at the end, then press **ENTER**.

**A2. Process (any mode)**
```bash
python3 heal_joint_replay.py process --name path1
```
Read the output: motion duration, peak joint speeds, suggested max `--speed`.
Open `data/path1_traj.png` (joints) and `data/path1_traj_ee.png` (the end-effector path, which should look like your shape). If it's wobbly, raise smoothing: `--smooth 0.2`. If a corner gets rounded too much, lower it: `--smooth 0.05`.

**A3. Back to heal_server.** Ctrl+C free_drive (support the arm), run `./heal_server` in T1, then relaunch in T2.

**A4. Replay, first at 0.3×**
```bash
python3 heal_joint_replay.py replay --name path1 --speed 0.3
```
1. It prints a **PRE-FLIGHT** (current vs start joints, approach time, path time). Type **y**.
2. The arm moves to the path start in a smooth joint-space motion. **Watch this move.** It goes from wherever the arm is (usually the end of the path) back to the start.
3. At the start it waits. Press **ENTER** to trace.
4. It traces, settles at the end, and stops by itself. **Ctrl+C at any time** sends zero velocity. The e-stop is still the real safety.

**A5. Check the result**
```bash
python3 heal_joint_replay.py analyze --name path1
```
This prints the RMS/max joint error and the **end-effector error in mm**, and saves `data/path1_replay_log_report.txt`, `..._joints.png` and `..._ee.png` (desired vs actual 3D path). These are your evidence for the report.

**A6. Iterate.** Try `--speed 0.5`, then 0.8, then 1.0.
- If the error grows with speed, try `--kp 3`.
- If it aborts with "tracking error", lower the speed or raise `--vmax` a little (default 0.4 rad/s).
- Keep the best run's files.

---

## 5. Branch B: end-effector recording only

**B1. Record (free drive).** Use the topic name from `check`. Add `--units mm` if check said millimetres.
```bash
python3 heal_ee_replay.py record --topic /THE_TOPIC --units mm --name path1
```
Same ENTER / hold still / trace / hold still / ENTER routine as Branch A.
**After stopping, do NOT move the arm.** Its current joints become the IK starting point and the frame check.

**B2. Process**
```bash
python3 heal_ee_replay.py process --name path1
```
Then look at `data/path1_ee_traj.png`.

**B3. Back to heal_server + launch** (as A3). The arm should still be at the end of the path.

**B4. Offline IK** (reads the live joints automatically)
```bash
python3 heal_ee_replay.py ik --name path1
```
- **FRAME CHECK** should be a few mm or less. That proves the recorded pose and the URDF agree on frame, units and tip point.
  - If it's large and you're sure the arm didn't move: wrong units (`--units`), or the vendor pose is for another link. Try `--tip-link` with a link from the printed list. As a last resort, `--offset auto` shifts the whole path to match.
  - If the recording has no orientation, IK holds the current tool orientation (`--orientation fixed`). Use `--orientation free` if many points fail.
- The report needs "samples with residual > 2 mm: 0" and no "branch" warning. Check `data/path1_ik_traj.png`.

**B5. Replay and analyze** (same options as Branch A)
```bash
python3 heal_ee_replay.py replay  --name path1 --speed 0.3
python3 heal_ee_replay.py analyze --name path1
```

---

## 6. Branch C: free drive gives nothing over ROS

Pick whichever fits the task.

**C1. Teach waypoints one by one.** Slow, but it is still "recorded in free drive".
For each waypoint:
1. In free drive, place the arm.
2. Stop free_drive, start `./heal_server` and the launch.
3. Capture the pose:
   ```bash
   python3 heal_joint_replay.py teach-waypoints --name path1 --append --single
   ```
4. Go back to free_drive for the next waypoint.

Then build a smooth trajectory through the waypoints (chained quintics, which don't stop at each point) and replay it:
```bash
python3 heal_joint_replay.py from-waypoints --name path1           # --stop-at-waypoints for sharp corners
python3 heal_joint_replay.py replay --name path1 --speed 0.5
```
Budget about 2 min per waypoint, so use 6–10 waypoints.

**C2. Path given as coordinates** (e.g. in the handout). Write a CSV with a header `x,y,z` (optionally `qx,qy,qz,qw`, or `roll,pitch,yaw`), in metres in `base_link`, then:
```bash
python3 heal_ee_replay.py import-csv --name path1 --file mypath.csv --ee-speed 0.03
python3 heal_ee_replay.py ik --name path1 --seed-at start
```
then replay as in B5.

**C3. Demo shape from the current pose:** circle, square, line or figure-8.
```bash
python3 heal_ee_replay.py make-shape --name circ --shape circle --size 0.05 --plane xy
python3 heal_ee_replay.py ik --name circ --seed-at start
python3 heal_ee_replay.py replay --name circ --speed 0.5
```

---

## 7. Finish and shut down

1. Make sure no script is running (they all stop themselves and send zeros).
2. Stop the launch (T2, Ctrl+C), then heal_server (T1, Ctrl+C), then whatever the lab procedure says for the arm.
3. **Copy `Group_3/data/` off the PC** (pen drive / scp / email). The `*_report.txt` and `*.png` files are your results.

---

## Safety rules (read once out loud before the first motion)

- E-stop holder has a hand on the e-stop for **every** motion command.
- The first replay of every new path runs at `--speed 0.3`.
- Clear the workspace. The approach move goes in a **straight line in joint space** from wherever the arm is to the path start, not along your path. If that could hit something (table, pen on paper), first move the arm close to the start (free drive, or `goto --q ...`).
- Only **one** thing publishes to `/velocity_controller/commands` at a time. Never run the starter homing / DLS scripts at the same time as these.
- The scripts abort and send zero velocity on: joint error > 0.2 rad (`--abort-err`), `/joint_states` silent for 0.25 s, or Ctrl+C. Speed is capped at `--vmax` (0.4 rad/s) and acceleration at `--amax` (1 rad/s²). All of this is software. The e-stop is the real safety.
- If the path involves a pen or tool touching a surface, record it with a few mm of clearance first, and only lower it once the replay looks right.
- Don't run `goto --home` unless you've checked the all-zeros pose is safe in your setup.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `rclpy not found` | That terminal isn't sourced (see §2) |
| `check` receives nothing | Wrong mode, launch not running, unsourced terminal, different `ROS_DOMAIN_ID`. Compare with `ros2 topic list` in the same terminal |
| Names don't contain joint1…6 | Use `--joints` with the real names (controller order) |
| Arm doesn't move, "nobody subscribes" warning | Velocity controller not active: check T2, run `ros2 control list_controllers` |
| Wrong joint moves | Command order ≠ joint1…6. Fix with `--joints` (§2 step 6) |
| ABORTED: tracking error | Too fast for the controller: lower `--speed`, raise `--kp` (2→3), or raise `--vmax` slightly |
| ABORTED: joint_states stale | Launch died or network hiccup. Restart the launch |
| Replay jerky | Raise `process --smooth` (0.15–0.25), or record slower and more steadily |
| "No motion found" in process | Recording too short or arm never moved: lower `--still-tol` or re-record |
| IK: big FRAME CHECK | Units (`--units mm`), wrong tip link (`--tip-link`), or the arm moved after recording |
| IK: residual > 2 mm / branch warning | Target unreachable with that orientation: use `--orientation free`, a smaller path, or check frames |
| "leaves the URDF joint limits" | Recorded path is too close to a limit. Re-record it a bit further from the limit |

## Parameter cheat sheet (replay / goto)

| Option | Default | Meaning |
|---|---|---|
| `--speed` | 0.5 | Path time scale (0.3 = 30 % of recorded speed). Auto-reduced if it would exceed 80 % of vmax |
| `--kp` | 2.0 | Position feedback gain (1/s). Higher = tighter but more aggressive |
| `--vmax` | 0.4 | Joint speed clamp (rad/s) |
| `--amax` | 1.0 | Joint acceleration limit (rad/s²) |
| `--approach-speed` | 0.15 | Peak joint speed for the move to the start (rad/s) |
| `--abort-err` | 0.2 | Abort if any joint is this far off (rad) |
| `--sim` | off | Simulated arm, no ROS needed |

## How the replay works (for your report)

1. **Recording:** timestamped joint angles (or end-effector poses) from free drive.
2. **Processing:** drop non-increasing timestamps, trim the still parts, resample to 100 Hz, zero-phase Gaussian smoothing, then velocities by central differences.
3. **(Branch B only) Offline IK:** damped least squares, `dq = Jᵀ(JJᵀ + λ²I)⁻¹e` with `e = [p* − p, log(R* Rᵀ)]`. Each sample is warm-started from the previous solution, so the arm stays on one elbow/wrist branch. The first sample is seeded from the robot's real joints.
4. **Approach:** quintic (rest-to-rest) joint-space move from the current pose to the path start, same math as the starter `QuinticJointSpacePlanner`.
5. **Tracking at 100 Hz:** `q̇_cmd = s·q̇_ref(s·t) + Kp (q_ref(s·t) − q)`. The first term is feedforward from the recording; the second is feedback that removes drift, since velocity commands integrate errors. The result is clamped and rate-limited, then published as `Float64MultiArray`.
6. **Evaluation:** joint error, plus end-effector error through forward kinematics of the URDF, desired vs actual.

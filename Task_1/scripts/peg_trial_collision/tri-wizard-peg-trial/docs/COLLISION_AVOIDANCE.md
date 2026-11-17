# Collision avoidance + grasp height — how it works

## 1. Why the gripper hit the other pegs

| Cause | Old behaviour | Now |
|---|---|---|
| Carry height | Lift to `pregrasp` height only → carried peg's bottom ~2 cm **below** the other peg tops → swept through them | Carry **over** every peg if the arm can reach that high, otherwise just above the board and steer **around** them (A*) |
| Transit path | One joint-space jump (arm swings in an arc, no checks) | Straight Cartesian segments, every 5–10 mm checked |
| Gripper tilt | `ik_best` allowed up to 20° tilt on approach/insert → fingers swing sideways | Gripper kept exactly vertical for every collision-checked move |
| Gripper opening | Fully open (≈86 mm finger span) | `pregrasp` = peg + ~10 mm; `release` = peg + ~8 mm |
| Order | Fixed order | Look-ahead search for an order where every peg is safe (placed pegs become obstacles) |
| Grasp depth 52 mm | Fingertips go 73 mm down among the neighbours; at insertion only ~2 mm above the board | Preferred 24 mm (see §3), lowered automatically only if needed |

## 2. Which method? (IK is not the problem)

IK only turns *one pose* into joint angles. Analytic, DLS or QP IK give the **same** pose,
so changing IK cannot fix collisions — the **path** between poses must change.

| Option | Verdict |
|---|---|
| Analytic IK (kept) | Exact and instant; used for every waypoint |
| DLS | Same answer, slower; no obstacle awareness by itself |
| QP IK with distance constraints | Possible, but needs the same distance checks, local minima, solver setup — overkill |
| MoveIt 2 + OMPL | Real motion planner, but heavy setup and the 4-DOF OMX needs position-only IK tricks |
| **Analytic IK + geometric collision model + A* (chosen)** | Simple, deterministic, explainable, ms-fast, uses the vision detections directly |

## 3. The "optimal" grasp height for 98 mm pegs

OMX fingertips are **21 mm below** the gripper centre (TCP); the thin finger ends 34 mm above it.

* Grip too **low** (old 52 mm): fingertips 73 mm down between the pegs; when the peg is pushed 20 mm
  into the socket the fingertips are ~2 mm from the board.
* Grip too **high** (< 15 mm): only the fingertip pads touch → the peg can tilt/slip.
* **24 mm below the top (default)**: ~45 mm of finger on the peg (secure), fingertips only 45 mm down,
  wide slider block stays 10 mm above the neighbours' tops, 52 mm of free peg below the fingers for insertion.

The planner starts at `grasp_depth` (24 mm) and goes lower, max `max_grasp_depth` (50 mm), only
when the arm cannot otherwise reach a safe height.

## 4. Robot height — the most important physical fix

With the OMX standing **on the table**, the gripper centre reaches at most ~13 cm high, so the fingertips
can only get above a 98 mm peg top in a narrow ring (~9–16 cm from the base; run `scripts/workspace_check.py`). Further out the planner
correctly refuses ("raise the robot"). Put the robot on a **firm 8–9 cm riser** (`table_z: -0.085`):
pegs are then carried *over* the others and most of the table becomes usable.
A 5 cm riser also works but every carry goes *around* (needs ≥ 2 cm gaps between pegs).

## 5. Before every run
```bash
ros2 topic echo --once --full-length /peg_trial/scene --field data > scene.json
python3 scripts/plan_check.py --scene scene.json     # order, grasp depth, route + out/plan_top_view.png
```
Tune in `task.yaml`: `table_z`, `collision.pregrasp_gap` (measure the finger gap!),
`robot.gripper_pregrasp` / `gripper_release`, `collision.safety_margin`.

## 6. Randomly placed pegs

Nothing is hard-coded: every run uses the positions **and rotations** vision measures.

* **Opening per peg:** a 55 mm square seen corner-first is ~55 mm wide across the fingers, face-first only ~39 mm.
  The planner computes the width along the finger axis for the measured yaw and opens to *that + 12 mm*
  (needs `robot.gripper_gap_open` / `gripper_gap_closed` measured, see task.yaml).
* **Squeeze grasp:** closing parallel fingers turns a polygon until a face lies flat on a finger, so after the
  grasp the peg's rotation relative to the arm is known whatever it was before (`grasp_geometry.py`).
* **Yaw at the socket:** the OMX has no wrist roll, so the remaining misalignment depends only on how the
  socket is turned relative to the robot. It is predicted and logged; it cannot be corrected by the arm.
  Mount the robot so the sockets' flat faces point roughly towards its base, and rely on socket clearance.
* **Order search** re-plans with the actual positions; pegs that touch or sit outside the reachable ring are
  reported, the rest are still placed.

Offline random-placement test (40 random layouts each, 55 mm pegs, fixed board):

| Robot | Min. centre spacing | All 4 pegs planned |
|---|---|---|
| 8.5 cm riser | 8.5 cm | 40 / 40 |
| 8.5 cm riser | 7 cm | 34 / 40 |
| 5 cm riser | 8.5 cm | 40 / 40 |
| on the table | 8.5 cm | 0 / 40 |

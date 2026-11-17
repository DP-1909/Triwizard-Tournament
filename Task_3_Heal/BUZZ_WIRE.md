# Buzz-wire task with `heal_buzz_wire.py` (updated for the real setup)

## What the robot does

1. Opens the gripper.
2. Comes down onto the hoop handle and closes.
3. **Flips:** swings the handle around the wire until it points straight at HEAL. The loop only spins around the wire; it does not move along it. This also squares up a hoop that was resting a bit crooked.
4. **Traces:** moves the loop along the wire, turning it at every bend so it stays perpendicular to the wire.
5. Stops at the right end, still holding the hoop.

**Why the handle points at HEAL:** that keeps the handle and gripper out of the wire's plane, so they never sweep through the neighbouring bends.

## What your new photos changed

- **The wire is smaller than first estimated.** The ends are 27 cm apart (30 cm is post to post), and the stretches are 3 cm apart. The bottoms are only about 1.7 cm above the board. `WIRE_CM` in the script now holds all 18 corners read off your annotated photo, to about ±3 mm.
- **The loop is a small triangle.** Its sides are 3.5, 2.5 and 2.5 cm, and the largest circle that fits inside has a radius of about 7 mm. Subtracting the wire and loop thickness leaves only **about 3.5 mm of free space** around the wire. Measurements can't guarantee that, so there's a **touch-up step** (step 6): the robot stops at every bend and you nudge the loop to the centre in 1 mm steps. The corrections are saved and used from then on.
- **The gripper body goes low.** The gripper turns with the loop, so on the vertical stretches it hangs to the side and its body reaches down toward the board. The planner checks this and tells you where the worst spot is. The two things that help most are covered in step 1.

---

## Step 1. Set up the hoop and measure a few things (10 min)

### Recommended start pose for the hoop

1. Put the loop around the wire in the **middle of the first U's bottom**. That's `START_AT = 2.5` in the script, which is the default.
2. Turn the handle so it points **straight at HEAL**, horizontally. Prop its far end on something small (an eraser or a block) so it stays level.
3. HEAL then grips **straight down from above**, near the far end of the handle, just before the ring terminal. That's about 10 cm from the loop centre.

This start pose does two things for you:
- The gripper ends up **above** the handle on the horizontal stretches.
- The gripper sits beyond the board's edge, over the lower mat.

If the TA insists on the photo's rest pose (loop in the bottom-left bend, handle leaning in the wire's plane), set `START_AT = 2`. Be aware that there's only about 3 cm of handle before the post, so the gripper will be close to the wire plane, and `plan` will probably warn about it.

### Measure these and edit the `EDIT HERE` block

| Setting | What it is | Default (from the photo) |
|---|---|---|
| `BOARD_TOP_CM` | height of the wooden board's top surface above the table HEAL stands on | 17.7 |
| `BOARD_FRONT_CM` | how far the wooden board extends from the wire toward HEAL | 8.0 |
| `MAT_TOP_CM` | height of the black mat beyond the board's edge | 15.7 |
| `GRIPPER_HALF_WIDTH_CM` | half the width across the closed fingers and linkages | 4.0 |
| `GRIPPER_HALF_THICK_CM` | half the gripper's thickness along the handle | 2.0 |
| `GRIPPER_BACK_CM` | distance from the finger pads back to the wrist | 15 |

Also check four or five `WIRE_CM` heights with a ruler: two bottoms, two tops and the right end. They're heights of the wire's centre above the table. If they're off by more than 3 mm, correct them.

**Hoop prep:** wrap tape around the handle where the fingers will grip, so it can't twist. Make sure the hoop's cable has slack all the way to the right end.

## Step 2. Bring-up

1. Start `heal_server`.
2. Start `velocity_control.launch.py`.
3. In a sourced terminal, `cd Group_3`.
4. Run:
   ```bash
   python3 heal_joint_replay.py save-urdf
   python3 heal_joint_replay.py goto --delta 0,0,0,0,0,0.1      # only the wrist turns; then -0.1 back
   python3 heal_buzz_wire.py where
   ```

`where` prints which way the tool's x, y and z axes point in the base frame. Use it to set these in the script:
- **`APPROACH_AXIS`**: the tool axis pointing out of the gripper, from the wrist toward the fingertips. To confirm, run `jog`, type `tz+ 10` and watch which way the gripper moves.
- **`FINGER_AXIS`**: the tool axis the fingers close along.
- **`TIP_TO_GRIP_CM`**: the distance from the URDF tip to the point between the finger pads. `where` prints how far the tip link is past the last joint:
  - if that's about the gripper's length, set it to 0;
  - if it's about 0, the tip link is at the flange: measure flange to pad centre and use that.

**Check `TIP_TO_GRIP_CM`:** in `jog`, type `rtz+ 20`, then `rtz- 20`. The point between the pads must stay put while the gripper turns.

## Step 3. Teach three poses with `jog` (15 min)

```bash
python3 heal_buzz_wire.py jog
```
Commands:
- `x+ 20`, `z- 5`: move along the base axes (mm).
- `tz+ 10`: move along the tool axes.
- `rtz+ 10`: rotate about a tool axis (degrees).
- `step 2`: set the default step.
- `open`, `close`: gripper.
- `where`: print the pose.
- `mark NAME`: save the pose.
- `q`: quit.

**`left` and `right` (the hoop isn't needed for these):**
1. Open the gripper, with the fingers pointing at the board and straddling the wire.
2. Put the point between the pads on the wire centre where the wire **leaves the left post**. Type `mark left`.
3. Using **only moves (no rotations)**, go to where the wire **enters the right post**. Type `mark right`.

**`grasp` (hoop in its start pose, see step 1):**
1. From above, open, lower the gripper over the handle about 1 cm before the ring terminal.
2. Line it up so the fingers close **across** the handle and the handle is centred between the pads.
3. Type `mark grasp`. **Don't close the gripper**, or you'll move the hoop.
4. Optionally, a few cm higher, type `mark pregrasp`.

The marks are saved in `data/buzz_marks.json`. Teach again only if the board or robot moves.

## Step 4. Plan (1 min)

```bash
python3 heal_buzz_wire.py plan
```

| Report line | Good | If not |
|---|---|---|
| CHECK span / CHECK height | within ±5 mm | re-teach `left`/`right` carefully, or use `--fit-span` |
| loop centre → gripper centre distance | ≈ where you gripped (≈ 10 cm) | `grasp` mark or `START_AT` is wrong |
| gripper on the UPPER side of the handle | `yes` | re-teach `grasp` from above (step 3) |
| gripper lowest clearance above board/mat | `OK` | see **Clearance** below |
| gripper closest to the wire plane | `OK` | grip further from the loop |
| joint limits / IK residual / WARNING | ok / ~0 / none | board out of reach: tell me the numbers |

Then open `data/buzz_plan.png`:
- **Left plot:** the red lines (the handle) all point at HEAL.
- **Right plot:** the wire (black) looks like your wire, the green star is where the hoop starts, and the blue bars (the loop) cross the wire at 90°.

If any check fails, `run` and `touchup` will ask you to type `yes` before moving.

## Step 5. Rehearse, then dry run (5 min)

```bash
python3 heal_buzz_wire.py run --sim
```
Then do a **dry run with the hoop removed**:
```bash
python3 heal_buzz_wire.py run
```
HEAL does the whole motion with an empty gripper. Watch the spot the clearance line names (for example "loop at u=12 cm, h=21 cm", which is the bottom of a vertical stretch). Make sure the gripper doesn't touch the board there.

## Step 6. Touch-up with the hoop (15 min). This is what makes it work.

Put the hoop back in its start pose, then:
```bash
python3 heal_buzz_wire.py touchup
```
1. It grasps and flips as in a run.
2. It moves to the **start**, then to **each bend** in turn, and stops.
3. At each stop, look at the loop from the side **and** from above. The wire should pass through the middle of the triangle.
4. Nudge the loop until it does:

| Type | Moves the loop |
|---|---|
| `h+` / `h-` | 1 mm up / down |
| `u+` / `u-` | 1 mm toward the right end / back |
| `n+` / `n-` | 1 mm toward HEAL / away |
| `h+ 3` | 3 mm (max 5 per nudge) |
| ENTER | it's centred: save this bend and go on |
| `q` | stop. Bends already confirmed stay saved; the current one does not |

Your nudges carry over to the next bend, so later bends usually need less. If you stop half-way, re-run `plan`, then `touchup` again: it starts over and adds to what's already saved.

When it's done:
```bash
python3 heal_buzz_wire.py plan
python3 heal_buzz_wire.py run
```
At the end it prints the **loop-centre deviation** (RMS / max in mm), which is your accuracy number. The log is saved in `data/`.

Useful options:
- `--speed 0.7`: slower.
- `--auto`: no ENTER between steps.
- `--release`: open the gripper at the end.
- `--no-gripper`: you operate the gripper yourself.

---

## Clearance: if the plan says the gripper may hit the board/mat

At the bottom of each vertical stretch, the gripper hangs sideways next to the handle. Its width then points up and down, about 4 cm each way, while the wire there is only about 3 cm above the board. In order of effect:

1. **Grip near the far end of the handle** (about 10 cm from the loop), so the gripper is beyond the board's edge and over the lower mat. In testing this took the worst spot from −2.1 cm to about 0.
2. **Put your measured values in the script.** The defaults for the gripper size, `BOARD_FRONT_CM` and `MAT_TOP_CM` are guesses. A slimmer gripper or a board that sticks out less toward HEAL may already make it OK.
3. If it's still slightly negative, **watch that spot in the dry run**, slowly (`--speed 0.5`), with the e-stop. The model is approximate.
4. If the TA allows it, raising the board on a flat spacer of thickness X helps a lot. Then add X to every `WIRE_CM` height and to `BOARD_TOP_CM`, and re-teach `left`/`right`/`grasp`.

## If the loop touches during the final run

- **It touched at a bend:** run `touchup` again; the new nudges add to the saved ones.
- **It touched between two bends:** add a `WIRE_CM` point there (half-way along that stretch). That changes the corner numbers, so delete `data/buzz_corrections.json`, then run `plan` and `touchup` again.
- **The deviation it reports is more than about 1 mm:** the robot lagged, so try `run --speed 0.7`.

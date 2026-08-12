# 5. Safety

A teleoperated robot has a human in the loop who stops when something looks
wrong. An autonomous one does not. A policy that has never seen the situation in
front of it will still produce an answer, with total confidence, and that answer
can be "drive into the table at full speed".

This is not a page about being nervous. It is a page about the specific things
that can go wrong here, and what stops each of them.

---

## Never run the robot alone

An adult who knows this robot is in the room, every time it moves. No exceptions,
not even for "just a quick test".

---

## Before every single run

- [ ] **Emergency stop within reach of your hand** — not across the desk. Know
      what it feels like without looking.
- [ ] **The area around the arm is clear** — no mugs, no laptops, no cables in
      the workspace, nobody's hands inside the reach of the arm.
- [ ] **You are not in the way.** Stand where the arm cannot reach you.
- [ ] **The gripper is empty** unless the test needs it not to be.
- [ ] **You ran with `--dry-run` first** and the printed numbers looked sane.
- [ ] **You know how to stop the script**: space to pause the policy, `q` to
      quit, Ctrl-C as the backup, e-stop as the real backup.

---

## The safety box

`robot_ai/arm.py` refuses to move the hand outside:

```
x   0.25 .. 0.58 m      away from the base
y  -0.40 .. 0.40 m      left / right
z   0.17 .. 0.60 m      up / down
```

These are the same limits the teleop script used while you were recording, which
means your demonstrations never went outside them, which means **the policy has
never seen what is out there.** The box is not decoration.

Do not widen it to see what happens. If you think a limit is wrong, say so and
change it together with an adult, once, deliberately.

`arm.times_clamped` counts how often the box had to intervene. A big number
during a run means the policy was pushing against a wall the whole time. That is
worth investigating, not ignoring.

---

## The gripper keeps going

The gripper motor is told a *speed*, not a position. It does what it was last
told until it is told something else. If your script stops sending commands
while the last one was `+1`, the motor keeps turning.

- always `gripper.set(0)` when you want it to stop
- always use `with Gripper() as gripper:` so it stops on the way out
- never put your fingers in it, even to test — use a pen

---

## Torque mode

While your script has the arm, the joints are in **torque control**: the arm is
not stiff, it is being actively held up by a calculation. If that calculation is
interrupted badly the arm can sag. This is why:

- `with Arm() as arm:` hands control back on any exit, including a crash
- you never `kill -9` a running deploy script; press `q`, or Ctrl-C
- if a script did die badly, the arm may need power-cycling before the next run

---

## When something goes wrong

1. **E-stop.** First. Not after you have thought about it.
2. Then tell an adult what happened, before restarting anything.
3. Then work out why. `arm.times_clamped`, the printed actions and
   `prediction.png` are usually enough to reconstruct it.

Breaking something is recoverable and everybody does it eventually. Hiding what
happened is what turns a small problem into a big one.

---

## And one thing that is not about safety, but matters

When it works — when the arm reaches for the cube on its own, using nothing but a
camera and half an hour of your demonstrations — take a moment. You did not
program that. You showed it, and it worked it out. That is a genuinely strange
and wonderful thing, and it stops being strange surprisingly quickly.

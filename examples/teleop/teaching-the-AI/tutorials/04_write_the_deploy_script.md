# 4. Write the deploy script

Now you swap yourself out for the policy.

Read `05_safety_checklist.md` **before** you run anything from this page on the
real robot. Not after.

---

## The idea

Your teleop recording script did this, a hundred times a second:

```
read the SpaceMouse  ->  nudge the hand  ->  press the buttons  ->  repeat
```

Your deploy script does the same thing, ten times a second, with one substitution:

```
take a picture  ->  ask the policy  ->  nudge the hand  ->  set the gripper  ->  repeat
```

The policy outputs the exact same kind of numbers the SpaceMouse did. The arm
cannot tell that a human is no longer holding it. That is the entire idea, and
it is worth appreciating how small the change is.

---

## What your script has to do

You are writing `deploy.py`.

### Step 1 — load the policy

```python
policy = Policy.load("my_policy.pt", device=pick_device())
```

Print `policy.chunk` and `policy.action_size`. You are about to rely on both.

### Step 2 — open the hardware, safely

```python
with Camera(dry_run=DRY) as camera, Arm(dry_run=DRY) as arm, \
     Gripper(dry_run=DRY) as gripper, Keys() as keys:
```

Using `with` is not a style preference here. It guarantees that if your code
crashes — or you hit Ctrl-C, or the policy produces a NaN — the arm still gets
handed back to its own safety controller and the gripper still stops turning.
Without it, a crash leaves a live robot.

Make `DRY` a command-line flag. **Your first ten runs are with `DRY = True`.**

### Step 3 — wait for a keypress before moving

Do not start the moment the script launches. Print the controls and wait:

```
[space] start / stop the policy
[q]     quit
```

Use `keys.pressed()` (it never blocks) and track a `running` flag. The robot must
only move while `running` is true, so that space is a real stop button.

### Step 4 — the loop

While not quitting:

1. `image = camera.get_image()` — skip this turn if it returns `None`
2. if your policy uses the state (`policy.state_size` is not 0), build those eight
   numbers **in the same order the recording stored them**: the seven joints from
   `arm.joint_positions()`, then `gripper.position()`. Get the order wrong and the
   policy is being told nonsense with total confidence
3. `actions = policy.predict(image, state)` → shape `(chunk, 7)`
4. **decide what to do with those `chunk` rows.** See below; this is the one real
   design decision in the whole script.
5. for the row you are using: the first three numbers go to
   `arm.move_like_joystick(row[0:3], seconds=0.1)`
6. the last number is the gripper. It is a decimal like `0.07`, and
   `gripper.set()` demands exactly `-1`, `0` or `+1`. You choose the cut-off.
7. print something every step — the action, the hand position — so you can see
   what it is thinking

### Step 5 — stop cleanly

On `q`, or on Ctrl-C, or on any exception: stop the gripper, let go of the arm.
If you used `with` in step 2, this is already done for you. Check that it is.

---

## The design decision: what to do with a chunk

The policy hands you a whole second of future — 10 rows. You have three choices,
and they behave very differently:

**A. Use only the first row, then throw the rest away and predict again.**
Reacts fastest to what it sees. But two consecutive pictures are nearly identical
and the policy's answers can differ, so the hand can shudder.

**B. Execute all 10 rows, then take a new picture.**
Smooth, committed movement. But for a whole second the robot is blind — it is
replaying a plan made from an old picture. If the cube moves, it does not notice.

**C. Something in between:** use the first 3 rows, then re-predict.

Try A and B for real, on the robot, and write down what each looks like. This is
a genuine trade-off in robotics — reaction time against smoothness — and you will
understand it far better from two minutes of watching than from any explanation.

> **Question 1.** Your policy was trained on moments sampled 10 times a second,
> but the arm's controller runs 100 times a second. `move_like_joystick(...,
> seconds=0.1)` holds one action for 10 of those ticks. What are you assuming
> about what the human's hand did in between? When would that assumption break?

---

## Dry run first

```bash
python deploy.py --dry-run
```

Nothing moves. The camera returns flat grey pictures, the arm only prints, the
gripper does nothing. What you are checking:

- does the loop run at about 10 times a second? (time it)
- do the printed actions look sane — between about -1 and +1?
- does space start and stop it? does q quit?
- if you Ctrl-C in the middle, does it exit cleanly with no traceback?

Only when all four are true do you touch the robot.

---

## Then, for real

Check the hardware first:

```bash
python tools/test_arm.py          # the arm moves ~5 cm and comes back
python tools/test_gripper.py      # which direction opens?
```

Then, with the checklist from tutorial 5 done, run for real. Hand on the e-stop.
First run: hold space for **one second only**, then stop and look at what
happened.

---

## What "working" looks like, and what it does not

It will not look like your teleoperation. Expect:

- movement that is roughly right and a bit hesitant
- the arm drifting toward where the cube *usually* was, if it cannot see this one
- a gripper that fires late, or not at all (remember tutorial 1, question 2)
- it doing well from starting positions like the ones you recorded, and getting
  lost from ones you never showed it

> **Question 2.** Put the cube somewhere you never put it during recording. What
> happens? Was that predictable from `check_policy.py`? (Be honest — could any
> number computed from your recordings have warned you?)

> **Question 3.** `arm.times_clamped` counts how often the safety box had to stop
> the hand. Print it at the end. If it is large, what is your policy trying to do,
> and what does that tell you about your demonstrations?

---

## If it goes wrong

| what you see | most likely cause |
|---|---|
| arm shoots to one side and stays there | policy predicting a constant — check `check_policy.py`, or you forgot to unnormalize (`predict` does it for you; do not do it twice) |
| tiny shaking, no progress | option A above, plus a policy that is unsure. Try option B |
| nothing moves at all | `running` flag never became true, or the actions are ~0. Print them |
| crashes with "this policy also feels 8 numbers" | you called `predict(image)` on a policy that wants the state too |
| moves fine, never grips | gripper cut-off too strict. Print the raw gripper number and look at its actual range |
| gripper never stops turning | you stopped sending commands instead of sending `set(0)` |

---

Last page: `05_safety_checklist.md`. Read it now if you have not.

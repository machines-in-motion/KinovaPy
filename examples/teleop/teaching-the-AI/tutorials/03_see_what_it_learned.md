# 3. Find out what it actually learned

Your training loss went down. That feels like success. It is not evidence of
anything yet — a policy can drive its training loss to nearly zero by memorising
18,000 pictures and still be useless on the 18,001st.

---

## Step 1 — mark your homework

```bash
python tools/check_policy.py --policy my_policy.pt
```

Use the **same** `--val-fraction` and `--seed` you used in `train.py`, or you will
be testing on takes the policy trained on, and the answer will be a happy lie.

You get a table like this, one row per action number:

```
action number         your error   lazy guess      verdict
control.0                  ...          ...        learned
...
```

**"Lazy guess"** is a policy that ignores the picture entirely and always answers
the average action. It is the dumbest thing that is not actually broken, and it
is the bar you have to clear. Beating it means the picture genuinely told your
policy something.

A policy trained the way tutorial 2 describes scores around **1.45x** overall,
with five of the six `control` numbers marked `learned`, `gripper_target.0`
marked `no better`, and often `control.4` marked `WORSE`. If you are in that
neighbourhood, you did it right.

> **Question 1.** Which action numbers did you beat the lazy guess on? Which not?
> Look back at tutorial 1, question 2 — is `gripper_target.0` one of the ones you
> beat? Why is that number so hard?
no, i couldn't beat it, maybe because it's a toggle, and there isn't a set zero point.

> **Question 2.** Remember that `control.3`, `control.4` and `control.5` do
> nothing to the robot. Does your policy predict them well or badly? Does it
> matter?
they predict some of them better, but it really doesn't matter, because those values aren't effecting the movment of the arm.
---

## Step 2 — look at the picture

`check_policy.py` writes `prediction.png`: your policy's answer (line) drawn on
top of what the human actually did (dots), for a take it has never seen.

This tells you far more than any single number. Look for:

- **Does the line follow the general shape of the dots?** Then it has learned the
  task, and the remaining error is detail. On a policy trained the way tutorial 2
  describes, `control.1` and `control.2` look like this — genuinely tracking.
- **Is one row a completely flat line at zero while the dots spike to ±1?** Look
  at the bottom row, `gripper_target.0`. That is what "learned nothing" looks
  like, and you predicted it in tutorial 1.
- **Is the line flat near the average while the dots swing around?** Then it has
  learned "when unsure, do the average" — it beat the lazy guess by a hair and
  understood nothing. Usually means: too little training, or too small a policy,
  or a learning rate so high the weights never settled.
- **Is the line jittery when the dots are smooth?** It is over-reacting to
  irrelevant details in the picture.
- **Does it get the beginning right and the end wrong?** Interesting — what is
  different about the end of a take?

---

## Step 3 — the two curves

Go back to the training and validation losses your script printed.

| what you see | what it means | what to do |
|---|---|---|
| both still falling | it is still learning | train longer |
| training falls, validation flattens | it is starting to memorise | this is the moment to stop |
| training falls, validation **rises** | it is now memorising in earnest | you trained too long; use the earlier saved copy |
| neither moves | something is broken | check normalization, learning rate, `optimizer.step()` |

That second row has a name: **overfitting**. It is not a mistake you can avoid
entirely, it is a point in time you have to notice.

> **Question 3.** At roughly which step did your validation loss stop improving?
> How much longer did you keep training after that? Was it wasted?
it stopped improving afte step 7, so 4 steps were wasted.

> **Question 4.** You have 132 takes. If you had 1,320, where would you expect
> that "stop here" point to move to, and why?
it would move to around 70, because i assume it improves relative to the amount of data.

---

## Step 3b — the experiment: does feeling itself actually help?

Your policy gets eight numbers about the robot as well as the picture. That
*sounds* obviously useful — a camera cannot always see whether the gripper is
open, a joint angle never lies.

Do not take anyone's word for it, including this page. Train a second policy that
is identical except for `state_size=0`, and run `check_policy.py` on both.

Three outcomes, and all three teach you something:

- **The state version is clearly better.** The picture was missing something the
  joints supply.
- **They are the same.** Whatever the joints told it, the picture already did.
- **The state version is worse on held-out takes while its training loss is
  lower.** The most interesting result: those eight numbers gave it a shortcut to
  memorise with, rather than something to reason from.

> **Question 5.** Whatever you get, explain it in two sentences. Then ask the
> harder question: this test compares predictions against a recording. Could a
> policy score identically here and behave differently on the real robot? Think
> about where the joint numbers come from during a real run, and how they are
> affected by what the policy did a moment ago.

## Step 4 — decide

You need a policy that is honestly better than the lazy guess — `check_policy.py`
prints an overall factor. Below about 1.1, do not put it on the robot; it will
wander. Above 1.5 it has genuinely learned the task and is worth trying.

If you are not there yet, in rough order of what to try:

1. train longer (cheapest thing to try, and often enough)
2. check the normalization mistake in tutorial 2 step 6 — again, honestly
3. try `chunk=20` instead of 10, or `chunk=5`
4. lower the learning rate by 10x and train longer
5. record more demonstrations (the real answer, and the most work)

Do **one** at a time and write down what happened. Changing three things and
getting a better number teaches you nothing about which one helped.

---

Next: `04_write_the_deploy_script.md` — the robot moves.

# 2. Write the training script

This is your job. Nobody is going to give you the code.

You are writing one file — call it `train.py` — that reads your prepared data,
teaches a `Policy` to copy what you did, and saves it to `my_policy.pt`.

If you have watched a "train a neural network in PyTorch" video, you have already
seen the important part. It is always these five lines, in a loop:

```
prediction = model(input)          # guess
loss = how_wrong(prediction, truth)  # measure how wrong the guess was
loss.backward()                    # work out which way each weight should move
optimizer.step()                   # move them a little
optimizer.zero_grad()              # forget the last batch's directions
```

Everything else in your script is plumbing around those five lines. Really.

---

## What your script has to do

### Step 1 — decide which takes are for practice and which are for the exam

```python
train_eps, val_eps = split_episodes(data_folder, val_fraction=0.15, seed=0)
```

You hold some takes back and never train on them, so that later you can ask an
honest question: *does this work on something it has never seen?*

Notice it splits by **take**, not by moment. Ask yourself why splitting by moment
would be cheating. (Hint: how different is frame 41 from frame 42?)

Print how many takes are in each group.

### Step 2 — build two datasets

```python
train_data = RobotData(data_folder, chunk=10, episodes=train_eps)
val_data   = RobotData(data_folder, chunk=10, episodes=val_eps)
```

`chunk=10` means each example asks for the next 10 moments = **1 second** of
movement. Print `len(train_data)` and `len(val_data)`.

Each example is a dictionary with three things in it: `"image"` (the picture),
`"state"` (where the joints and the gripper are), and `"actions"` (what to
predict). Print the shapes of all three for `train_data[0]` and make sure they
are what you expect before going any further.

### Step 3 — wrap them in DataLoaders

A `torch.utils.data.DataLoader` does three jobs: it groups examples into batches,
it shuffles them, and it loads the next batch on other CPU cores while the GPU
works on the current one.

- batch size: start with 32
- shuffle: **yes** for training, **no** for validation — and be ready to explain why
- `num_workers=4` (this is the "load on other cores" part; without it your GPU
  spends most of its time waiting for JPEGs)

> Common mistake: forgetting `shuffle=True`. Your data is in time order, so
> without shuffling every batch is one second of one take, all nearly identical.
> The network will lurch around following whatever it saw most recently.

### Step 4 — build the policy

```python
mean, std = train_data.action_stats()
state_mean, state_std = train_data.state_stats()
policy = Policy(action_size=train_data.action_size, chunk=10,
                action_mean=mean, action_std=std,
                state_size=train_data.state_size,
                state_mean=state_mean, state_std=state_std)
policy.to(device)
```

Take the statistics from the **training** data only, never from the validation
data. If your exam paper influences your revision, the exam is no longer a test.

Passing `state_size=0` builds a policy that works from the picture alone and
ignores the joints entirely. You will want that later — see tutorial 3 — so make
it easy to switch.

Print how many weights the policy has. (`sum(p.numel() for p in policy.parameters())`
— it should be tens of millions.)

### Step 5 — pick an optimizer

`torch.optim.AdamW(policy.parameters(), lr=1e-4)` is a sensible starting point.

The learning rate is the size of the step taken at `optimizer.step()`. Too big
and the loss jumps around and never settles; too small and you wait forever. You
will get to try both.

### Step 6 — the training loop

For a few thousand steps, or for a number of passes over the data ("epochs"):

1. take the next batch from the DataLoader
2. move `batch["image"]`, `batch["state"]` and `batch["actions"]` to the device
3. run them through the policy → `policy(images, state)` → the guess
   (if you built the policy with `state_size=0`, pass `None` instead)
4. **normalize the true actions** with `policy.normalize_actions(...)` before
   comparing, because the policy works in normalized units
5. compute the loss: `torch.nn.functional.l1_loss(guess, truth)`
6. `backward`, `step`, `zero_grad`
7. print the loss every 50 steps or so

> **The mistake almost everyone makes:** comparing the policy's output against
> *raw* actions instead of normalized ones. Nothing crashes, and here is the nasty
> part — your loss looks **better**, not worse. Raw joystick numbers are small, so
> a loss of 0.1 appears where a correct run shows 0.45, and it is tempting to
> believe you did something clever.
>
> You have not. The policy is now producing raw-sized numbers, and `predict()`
> will faithfully multiply them by the spread and add the average on the way out,
> so the robot gets nonsense. `check_policy.py` catches it — every row comes back
> `WORSE`. If your loss is suspiciously low and your check is terrible, this is
> why.

Why L1 (average absolute error) rather than L2 (squared error)? L2 punishes big
mistakes enormously, so when a picture has two valid answers it pulls hard toward
the middle of them. L1 does that less. Given what you read in tutorial 1 about
the same picture having more than one right answer, that matters here.

### Step 7 — check on the validation takes every so often

Every few hundred steps, stop training and run through `val_loader` with
`torch.no_grad()`, computing the same loss but **not** calling backward or step.
Print it next to the training loss.

This number is the one that actually matters. Watch what the two do relative to
each other over time — you will use that in tutorial 3.

### Step 8 — save

```python
policy.save("my_policy.pt")
```

Save at the end, and ideally every time the validation loss reaches a new best.
The file contains the weights *and* the units, so nothing else is needed to use
it later.

### Step 9 — make it repeatable

Set `torch.manual_seed(0)` at the top so two runs of the same script give the
same result. Otherwise you cannot tell whether a change you made helped or
whether you just got lucky.

---

## What you should see

These are real numbers, measured on your machine (RTX 2080 Ti, batch size 32,
`chunk=10`, `lr=1e-4`, with the state). Yours will not match exactly, but they
should be close.

| | |
|---|---|
| speed | about **20 batches per second** |
| one full pass over the data ("epoch") | about 490 steps, roughly **25 seconds** |
| your split | 112 takes to train on, 20 held back |
| loss at the very first step | about **0.7** — that is what "no idea at all" looks like |
| after 1 epoch | training **0.43**, validation **0.46** |
| after ~12 epochs (6000 steps) | training **0.23**, validation **0.46** |
| when the validation stops improving | **surprisingly early — around 4 epochs** |

That last row is not a mistake, and it is the most interesting thing on this
page. The training loss keeps falling for hours; the validation loss is done
almost immediately. Tutorial 3 is about what that means.

If your loss starts at 40, or at 0.001, something is wrong with the normalization
— go back to step 6.

If your loss is completely flat from the first step, check that you called
`optimizer.step()`, that the loss came from the policy's own output, and that
you did not accidentally wrap the training in `torch.no_grad()`.

---

## When it runs

```bash
python train.py
python tools/check_policy.py --policy my_policy.pt
```

Then go to `03_see_what_it_learned.md`.

---

## Stretch goals, once it works

- Train a second policy with `chunk=1` (predict only the very next moment) and
  compare. Which is better on the robot? Why might that be?
- Try `lr=1e-3` and `lr=1e-5`. Draw the two loss curves. Explain the shapes.
- Freeze the ResNet (`for p in policy.backbone.parameters(): p.requires_grad = False`)
  and train only the head. Faster? Worse? By how much?
- Flip every picture left-to-right at random during training. Would that help
  here, or actively hurt? (Careful — think about what "move left" means in a
  mirrored picture.)

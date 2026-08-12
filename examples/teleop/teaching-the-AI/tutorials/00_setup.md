# 0. Getting the computer ready

About 15 minutes of typing, then a quarter of an hour of waiting while the
computer unpacks your recording.

---

## Step 1 — open a terminal in this folder

```bash
cd ~/teaching-the-AI          # or wherever this folder ended up
source activate_env.sh
```

`source`, not `python`. The difference matters: a normal script gets its own
little world and throws it away when it finishes, so it cannot change *your*
terminal. `source` runs the lines as if you had typed them yourself.

You should see:

```
environment ready:  kinova
next:               python check_setup.py
```

**You have to do this in every new terminal window.** If a command suddenly
says `ModuleNotFoundError: No module named 'torch'`, this is almost always why.

<details>
<summary>What that script actually does, if you are curious</summary>

Two things. It switches on the `kinova` conda environment, which is the folder
containing the right Python and the right libraries. And it sets
`LD_LIBRARY_PATH`, which fixes an error you would otherwise hit later:

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

That looks like a broken installation but is not. Some robot libraries were
built against a newer version of a system library than Ubuntu ships. Both copies
are on the computer; `LD_LIBRARY_PATH` just says which folder to look in first.
</details>

---

## Step 2 — check everything is there

```bash
python check_setup.py
```

You will get a list of `ok` and `!!` lines. Read it, do not skim it.

- Anything marked **FAIL** must be fixed before you can continue. Ask.
- Anything marked **!!** is only needed later. If the arm, camera or gripper are
  switched off right now, they will show `!!` and that is completely fine — you
  do not need the robot to train a policy.

You **do** need, right now: `python`, `numpy`, `opencv`, `torch`, `torchvision`,
`lerobot`, and `robot_ai blocks`.

> **Question 1.** Does it say you have a GPU? Write down which one. Later, when
> training takes 4 minutes instead of 4 hours, that is why.

---

## Step 3 — turn your recording into training data

Your recording is stored as compressed video. Video is wonderful for storage and
terrible for training: every single picture has to be decoded before it can be
used, which takes about 30 milliseconds. Training will look at each picture
dozens of times. Do the arithmetic — that is hours of the computer doing nothing
but unzipping.

So we unpack it once, into plain small JPEGs:

```bash
python prepare_data.py
```

This takes **about 12 minutes** for 132 takes, and writes about 280 MB. It prints
its progress and an estimate as it goes. Let it finish — go and read tutorial 1
while you wait.

When it is done you have:

```
data/pick-cube/
    frames/000/00000.jpg ...     one small picture per moment
    actions.npy                  what the human did
    states.npy                   what the robot felt (7 joints + the gripper)
    episode_index.npy            which take each moment came from
    frame_index.npy              how far into that take
    meta.json                    the names and sizes
```

> **Question 2.** Open `data/pick-cube/meta.json`. It lists 7 action names. Your
> recording contained 14 numbers per moment. Read the top of `prepare_data.py`
> and write down, in your own words, why we threw 7 of them away.

> **Question 3.** Look at one of the JPEGs. It is 320x180. The camera records
> 640x480. Why would making the picture *bigger* not necessarily make the policy
> better — and what does it definitely cost?

> **Question 4.** `states.npy` holds 8 numbers per moment: the 7 joint angles and
> the gripper. The recording actually contains 15. Read the top of
> `prepare_data.py` to find out which 7 were left out and why. One of them is
> `time` — explain, in your own words, why letting the policy see the clock would
> make it look brilliant on your recordings and useless on the robot.

---

## Step 4 — check it again

```bash
python check_setup.py
```

`prepared data` should now say `ok` with your number of moments and takes.

---

Next: `01_look_at_the_data.md`

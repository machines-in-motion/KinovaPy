# Teaching the AI

You have already done the hard part: you drove the robot by hand, over and over,
and recorded what you did. **132 takes. About half an hour of your life. 18,344
moments, each one a picture plus the movement you were making.**

Now you are going to teach a neural network to do it instead of you, and then
hand it the controls.

That idea has a name — **behaviour cloning** — and it is exactly as literal as it
sounds. You are not going to tell the robot what a cube is, where the table is,
or what "pick it up" means. You are going to show it thousands of pictures next
to the movement a human made when looking at that picture, and let it work out
the connection by itself.

---

## The plan

| | what you do | where |
|---|---|---|
| 0 | get the computer ready | `tutorials/00_setup.md` |
| 1 | look hard at your own data | `tutorials/01_look_at_the_data.md` |
| 2 | **write the training script** | `tutorials/02_write_the_training_script.md` |
| 3 | find out what it actually learned | `tutorials/03_see_what_it_learned.md` |
| 4 | **write the deploy script** | `tutorials/04_write_the_deploy_script.md` |
| 5 | the safety rules, before the robot moves | `tutorials/05_safety_checklist.md` |

Steps 2 and 4 are yours to write. Everything else is done for you or is just
running a command. Nobody is going to hand you the answer — that is the point.

---

## The lego blocks

You never have to touch the robot's control theory, the video decoder, the CAN
bus or the optimiser. You get eight pieces, and you snap them together.

### Learning

```python
from robot_ai import RobotData, split_episodes, Policy, pick_device
```

```python
data = RobotData(folder, chunk=10, episodes=None)
    len(data)                  how many training examples
    data[i]                    {"image":   (3,180,320) tensor,
                                "state":   (8,) tensor      <- only if the data has it
                                "actions": (chunk,7) tensor}
    data.action_stats()        -> (average, spread), one number per action
    data.state_stats()         -> the same, for what the robot feels
    data.action_names          ['control.0', ..., 'gripper_target.0']
    data.state_names           ['joints.0', ..., 'joints.6', 'gripper.0']
    data.action_size           7
    data.state_size            8   (0 if you prepared the data without state)
    data.episodes              which takes are in here

train_eps, val_eps = split_episodes(folder, val_fraction=0.15, seed=0)

policy = Policy(action_size=7, chunk=10, action_mean=..., action_std=...,
                state_size=8, state_mean=..., state_std=...)   # state_size=0 -> pictures only
    policy(images, state)             (B,3,180,320),(B,8) -> (B,chunk,7)  in NETWORK units
    policy.normalize_actions(a)       real units -> network units
    policy.unnormalize_actions(a)     network units -> real units
    policy.predict(image, state)      one picture (+ state) -> (chunk,7) in REAL units
    policy.state_size                 how many numbers it expects to be told
    policy.save(path)
    Policy.load(path, device)

pick_device()                  'cuda' if you have a GPU, else 'cpu'
```

### Driving

```python
from robot_ai import Camera, Arm, Gripper, Keys
```

```python
with Camera() as camera:
    image = camera.get_image()        # (180,320,3), exactly like the training pictures

with Arm() as arm:                    # homes itself, takes control, lets go on exit
    arm.move_like_joystick([x,y,z], seconds=0.1)   # each number -1..+1, like the SpaceMouse
    arm.hand_position()               # (x,y,z) in metres
    arm.joint_positions()             # the 7 joint angles - the first 7 numbers of `state`
    arm.times_clamped                 # how often the safety box had to step in

with Gripper() as gripper:
    gripper.set(+1)                   # +1 / 0 / -1, same as `gripper_target` in your data
    gripper.position()                # how far it has turned - the 8th number of `state`

with Keys() as keys:
    keys.pressed()                    # every key since you last asked, e.g. ' q'
```

Every driving block takes `dry_run=True`, which makes it pretend. **Write your
deploy script in dry-run first.** A bug that prints a wrong number costs you
nothing; the same bug driving a 7 kg arm costs you a robot.

### Tools you can run any time

```bash
python check_setup.py                        # is this computer ready?
python prepare_data.py                       # recording -> training folder (run once, ~12 min)
python tools/show_data.py                    # look at the data
python tools/check_policy.py --policy ...    # mark your own homework
python tools/test_arm.py --dry-run           # does the arm block work?
python tools/test_gripper.py                 # which way does the gripper open?
```

---

## Three things that are true and worth knowing now

**The policy speaks joystick.** It does not output "move to x=0.4". It outputs
the same numbers your SpaceMouse was producing — three for the hand, one for the
gripper button. Your deploy script's job is to hand those numbers to the arm
exactly the way the teleop script handed it yours. The robot cannot tell the
difference, and that is the whole trick.

**The policy can also feel itself.** As well as the picture, it is told where its
seven joints are and how far the gripper has turned - eight numbers, the same
ones the recording stored. A camera cannot always see whether the gripper is
open; a joint angle never lies about it. Whether that actually helps is a
question you can answer with an experiment rather than an opinion, and tutorial 3
asks you to run it.

**The policy only knows what it saw.** Not that a cube exists. Not that the table
is at a certain height. It has seen 18,344 pictures and 18,344 answers, and it
will confidently produce an answer for a picture unlike any of them. Move the
camera 10 cm and it may fail completely. That is not a bug in your code — it is
what this method *is*, and knowing its edges is more useful than any trick.

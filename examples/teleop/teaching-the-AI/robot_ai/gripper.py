"""The gripper block: a GIM4305 motor on a CAN bus, doing exactly what the buttons did.

While recording, you did not tell the gripper an angle - you held a button and it
turned, and you let go and it stopped. The dataset stores that as one number per
moment, called `gripper_target`:

        +1   button 0 was held   (motor turns one way)
         0   nothing held        (motor holds still)
        -1   button 1 was held   (motor turns the other way)

So that is exactly what this block takes. `gripper.set(+1)` is "the human is
pressing button 0 right now".

TWO THINGS THAT WILL BITE YOU

1. The motor keeps doing the last thing it was told. If you send +1 and then
   stop sending anything, it keeps turning. Always send 0 when you want it to
   stop, and always call `stop()` at the end of your script.

2. Nobody wrote down which direction opens and which closes. Find out yourself
   with `python tools/test_gripper.py`, with nothing in the gripper, and write
   it in your notes.
"""

import time

# The five numbers the motor protocol wants. These are copied from the teleop
# script so the gripper behaves identically to the recording.
SPEED = 4.0  # how fast it turns (rad/s)
TORQUE = 0.05  # a small extra push (Nm)
STIFFNESS = 0.0  # 0 = "do not try to hold an angle", we only care about turning
DAMPING = 0.03


class Gripper:
    """The hand. Use it with `with`, so it always stops."""

    def __init__(self, dry_run=False):
        self.dry_run = bool(dry_run)
        self._motor = None
        self.last_direction = 0
        self.last_reply = None

    # --------------------------------------------------------------- lifetime

    def connect(self):
        if self.dry_run:
            print("[gripper] dry run: pretending. The motor will not turn.")
            return self
        from KinovaPy.gim4305 import Gim4305

        self._motor = Gim4305()
        self._motor.enable()
        time.sleep(0.1)
        # Zeroing here is what the recording script did too, so `position()`
        # counts from the same place the training data counted from.
        self._motor.set_zero()
        self.set(0)
        print("[gripper] ready")
        return self

    def stop(self):
        """Stop turning and release the bus. Never skip this."""
        if self.dry_run or self._motor is None:
            return
        try:
            self.set(0)
        finally:
            self._motor.close()
            self._motor = None
            print("[gripper] stopped")

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ---------------------------------------------------------------- moving

    def set(self, direction):
        """direction: +1, 0 or -1 - the same number as `gripper_target` in your data.

        Decimals are refused on purpose. Your policy predicts things like 0.34,
        and quietly rounding that to 0 would mean a gripper that never fires and
        never tells you why. Deciding where the cut-off goes is your job.
        """
        value = float(direction)
        if value not in (-1.0, 0.0, 1.0):
            raise ValueError(
                f"direction must be exactly -1, 0 or +1, got {direction!r}. "
                f"Your policy gives decimals - you have to decide where the cut-off is."
            )
        direction = int(value)
        self.last_direction = direction
        if self.dry_run:
            return None
        speed = SPEED * direction
        torque = TORQUE * direction
        self.last_reply = self._motor.command(
            place=0.0, zoom=speed, jk=STIFFNESS, jl=DAMPING, umph=torque
        )
        return self.last_reply

    def position(self, default=0.0):
        """Roughly how far the motor has turned, in radians.

        This is the number the recording stored as `gripper`, so it is what a
        policy trained with proprioception expects to be handed. Two warnings,
        both of which you can see in tools/show_data.py:

        * the driver ROUNDS it to whole radians, so it moves in big steps
        * it WRAPS: past +12.5 it reappears at -12.5, so 12 and -12 can be
          nearly the same physical position

        Before the first command there is nothing to report, so you get
        `default` - the same 0.0 the recording script used.
        """
        if self.last_reply is None:
            return default
        return self.last_reply["position"]

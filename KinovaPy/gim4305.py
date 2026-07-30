"""
gim4305.py  —  Our little library for talking to the GIM4305 motor.

This is the "brain" that the lessons build up and then use. A few of the most
important functions are left for YOU to finish (look for `# TODO`). You'll
complete them during Lessons 3 and 4. Until you do, they raise an error on
purpose so you can't forget.

Everything the motor understands is the classic "MIT / Mini-Cheetah" CAN
protocol. Don't just take our word for the numbers below — see the big warning
about the ranges.
"""

from . import config
import numpy as np
# Note: we import the `can` library *inside* the Gim4305 class below, not here.
# That way Lessons 3 and 4 (which only test the packing MATH) can run even
# before python-can is installed and with no motor plugged in.


# ===========================================================================
#  SECTION A — The number ranges (min/max) and how many bits each value gets
# ===========================================================================
#
#  ⚠️  IMPORTANT — VERIFY THESE AGAINST YOUR MOTOR'S DATASHEET!  ⚠️
#
#  These min/max values decide how your real numbers get squeezed into whole
#  numbers (see Lesson 3). If they don't match what your motor's firmware
#  expects, the motor will move to the WRONG angle or with the WRONG force.
#
#  The values below are typical defaults for this style of motor, but the
#  torque and velocity limits in particular vary between firmware versions.
#  Finding the correct ones in the datasheet and fixing them here is part of
#  the job of a real engineer. Treat this as your first "bug to hunt."
# ---------------------------------------------------------------------------

# Position: how far the output can rotate, in radians. (12.5 rad ≈ 716°)
P_MIN, P_MAX = -12.5, 12.5
# Velocity: rotation speed, in radians/second.
V_MIN, V_MAX = -30.0, 30.0
# Kp: position "stiffness" gain (0 = don't care about position).
KP_MIN, KP_MAX = 0.0, 500.0
# Kd: velocity "damping" gain (0 = don't care about speed).
KD_MIN, KD_MAX = 0.0, 5.0
# Torque: extra feed-forward twist, in newton-metres. <-- MOST likely to differ
T_MIN, T_MAX = -5.0, 5.0

# How many bits each value is allowed in the 8-byte message.
# Add them up: 16 + 12 + 12 + 12 + 12 = 64 bits = 8 bytes. It fits exactly!
POS_BITS = 16
VEL_BITS = 12
KP_BITS  = 12
KD_BITS  = 12
TOR_BITS = 12


# ===========================================================================
#  SECTION B — The three "magic" command frames (given to you, complete)
# ===========================================================================
# Seven 0xFF bytes, then one byte that picks the command. See README Part 2.
ENTER_MOTOR_MODE = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
EXIT_MOTOR_MODE  = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
SET_ZERO         = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]


# ===========================================================================
#  SECTION C — The quantization math  (⭐ YOU COMPLETE THIS IN LESSON 3)
# ===========================================================================

def float_to_uint(x, x_min, x_max, bits):
    fraction = (x-x_min)/(x_max-x_min)
    big = (2**bits-1)
    return int(np.floor(fraction*big))

def uint_to_float(n, x_min, x_max, bits):
    """
    The REVERSE of float_to_uint: turn the whole number `n` back into a real
    number between x_min and x_max. You'll need this in Lesson 4 to read what
    the motor reports back.

    Recipe (just undo the steps):
      1) span = x_max - x_min
      2) fraction = n / (2**bits - 1)
      3) return fraction * span + x_min
    """
    # TODO: replace the line below using the recipe above.
    span = (x_max - x_min)
    fraction = n/(2**bits-1)
    return round((fraction*span+x_min))


# ===========================================================================
#  SECTION D — Packing a command into 8 bytes  (⭐ YOU COMPLETE THIS IN LESSON 3)
# ===========================================================================

def pack_command(place, zoom, jk, jl, umph):
    """
    Take the five real numbers and build the 8-byte list to send to the motor.

    Step 1 — turn each real number into a whole number using float_to_uint()
             and the ranges/bits from Section A.
    Step 2 — stack them into 8 bytes following this map (README Part 4):

      Byte 0: position bits 15..8   (the top half of the 16-bit position)
      Byte 1: position bits 7..0    (the bottom half)
      Byte 2: velocity bits 11..4   (the top 8 of velocity's 12 bits)
      Byte 3: velocity bits 3..0  +  kp bits 11..8   (SHARED byte!)
      Byte 4: kp bits 7..0
      Byte 5: kd bits 11..4
      Byte 6: kd bits 3..0  +  torque bits 11..8     (SHARED byte!)
      Byte 7: torque bits 7..0

    The tools you need (see README cheat sheet):
      >>  shift right (grab the TOP bits of a number)
      &   mask       (keep only the bits you want, e.g. & 0xFF keeps low 8)
      <<  shift left (slide bits up to make room)
      |   or / merge (glue two half-bytes into one byte)

    A couple are done for you as worked examples. Finish the rest.
    """
    # Step 1: quantize. (These call the function YOU wrote above.)
    p = float_to_uint(place, P_MIN, P_MAX, POS_BITS)   # 16-bit
    v = float_to_uint(zoom, V_MIN, V_MAX, VEL_BITS)   # 12-bit
    kp_i = float_to_uint(jk,     KP_MIN, KP_MAX, KP_BITS) # 12-bit
    kd_i = float_to_uint(jl,     KD_MIN, KD_MAX, KD_BITS) # 12-bit
    t = float_to_uint(umph,   T_MIN, T_MAX, TOR_BITS)   # 12-bit

    data = [0] * 8

    data[0] = (p >> 8) & 0xFF          
    data[1] = p & 0xFF                 
    data[2] = (v >> 4) & 0xFF
    data[3] = ((v & 0xF) << 4) | (kp_i >> 8)
    data[4] = kp_i & 0xFF
    data[5] = (kd_i >> 4) & 0xFF
    data[6] = ((kd_i &0xF) << 4 | (t >> 8))
    data[7] = t & 0xFF

    return data


# ===========================================================================
#  SECTION E — Unpacking the motor's reply  (⭐ YOU COMPLETE THIS IN LESSON 4)
# ===========================================================================

def unpack_reply(data):
    """
    The motor answers with its OWN measured state, packed like this:

      Byte 0: the motor's CAN id
      Byte 1: position bits 15..8
      Byte 2: position bits 7..0
      Byte 3: velocity bits 11..4
      Byte 4: (velocity bits 3..0) in the high nibble + (current bits 11..8) in the low nibble
      Byte 5: current bits 7..0

    Return a dict: {"id":..., "position":..., "velocity":..., "current":...}
    with position/velocity turned back into real numbers via uint_to_float().
    (Current uses the torque range as a stand-in here.)

    Recipe:
      motor_id = data[0]
      p_int = (data[1] << 8) | data[2]                 # glue two bytes into 16 bits
      v_int = (data[3] << 4) | (data[4] >> 4)          # 8 bits + top nibble
      i_int = ((data[4] & 0xF) << 8) | data[5]         # low nibble + a byte
      then convert p_int and v_int with uint_to_float(...)
    """
    # TODO: build and return the dict following the recipe above.
    motor_id = data[0]
    p_int = (data[1] << 8) | data[2]                 
    v_int = (data[3] << 4) | (data[4] >> 4)          
    t_int = ((data[4] & 0xF) << 8) | data[5]
    p= uint_to_float(p_int, -12.5, 12.5, 16)#position
    v=uint_to_float(v_int, -30.0, 30.0, 12)#velocity
    t=uint_to_float(t_int, -5.0, 5.0, 12)#torque
    return  {"id": motor_id, "position": p, "velocity": v, "current": t}


# ===========================================================================
#  SECTION F — The bus wrapper (given to you — read it, it's how frames are sent)
# ===========================================================================

class Gim4305:
    """A small helper that opens the CAN bus and sends/receives frames."""

    def __init__(self, motor_id=config.MOTOR_ID):
        import can  # imported here so Lessons 3 & 4 don't need it installed
        self._can = can
        self.motor_id = motor_id
        # Open the CANable using the settings from config.py
        self.bus = can.interface.Bus(
            interface=config.CAN_INTERFACE,
            channel=config.CAN_CHANNEL,
            bitrate=config.CAN_BITRATE,
        )

    def _send(self, data):
        """Send 8 raw bytes to our motor's ID and (try to) read one reply."""
        msg = self._can.Message(
            arbitration_id=self.motor_id,
            data=data,
            is_extended_id=False,
        )
        self.bus.send(msg)
        reply = self.bus.recv(timeout=0.5)   # wait up to 0.5s for an answer
        return reply

    def enable(self):
        """Wake the motor up (enter motor mode)."""
        return self._send(ENTER_MOTOR_MODE)

    def disable(self):
        """Put the motor to sleep (exit motor mode)."""
        return self._send(EXIT_MOTOR_MODE)

    def set_zero(self):
        """Call the motor's current position '0'."""
        return self._send(SET_ZERO)

    def command(self, place, zoom, jk, jl, umph):
        """Send a full movement command and return the motor's reply (parsed)."""
        data = pack_command(place, zoom, jk, jl, umph)
        reply = self._send(data)
        if reply is None:
            return None
        return unpack_reply(list(reply.data))

    def close(self):
        """Always call this when you're done, so the bus is released cleanly."""
        try:
            self.disable()
        finally:
            self.bus.shutdown()

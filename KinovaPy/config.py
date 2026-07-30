"""
config.py  —  All the settings in ONE place.

You shouldn't need to change much here except CAN_CHANNEL/INTERFACE to match
how your CANable showed up in SETUP.md.

Nothing to complete in this file. Read it so you know what the knobs are.
"""

# ---------------------------------------------------------------------------
# How we talk to the CANable 2.0
# ---------------------------------------------------------------------------
# On Linux with candleLight firmware (recommended):
CAN_INTERFACE = "socketcan"    # the "backend" python-can uses
CAN_CHANNEL   = "can0"         # the name from `ip link ... up` in SETUP.md
CAN_BITRATE   = 1_000_000      # 1 Mbps — MUST match the motor

# --- On Windows/Mac (candleLight via gs_usb), comment the 3 lines above and
#     use these instead: ---
# CAN_INTERFACE = "gs_usb"
# CAN_CHANNEL   = 0            # first gs_usb device
# CAN_BITRATE   = 1_000_000

# --- On slcan/serial firmware, it looks more like: ---
# CAN_INTERFACE = "slcan"
# CAN_CHANNEL   = "/dev/ttyACM0"   # or "COM3" on Windows
# CAN_BITRATE   = 1_000_000


# ---------------------------------------------------------------------------
# Which motor we're talking to
# ---------------------------------------------------------------------------
MOTOR_ID = 1     # the GIM4305's default CAN ID. Change if you reconfigured it.

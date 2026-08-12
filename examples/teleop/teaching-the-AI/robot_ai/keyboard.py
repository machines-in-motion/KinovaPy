"""The keyboard block: read keypresses without ever stopping the robot loop.

`input()` is useless here - it freezes everything until you press Enter, and a
robot that freezes mid-motion is a robot doing something unplanned. This reads
whatever keys are waiting and returns immediately, every single loop.

Adapted from the teleop recording script (examples/teleop/button.py).
"""

import os
import select
import sys
import termios
import tty


class Keys:
    """Non-blocking keyboard. Use it with `with`, or your terminal stays weird."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self._fd = None
        self._old_attrs = None

    def start(self):
        try:
            self._fd = self.stream.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (termios.error, ValueError, AttributeError, OSError):
            # Not a real terminal (piped input, an IDE console, nohup...).
            # Pretend no key is ever pressed rather than crashing the run.
            self._fd = None
            self._old_attrs = None
            print("[keys] this is not a real terminal - keyboard control is off")
        return self

    def restore(self):
        if self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.restore()
        return False

    def pressed(self):
        """Every key pressed since you last asked, as a string ('' if none).

        Returns them all, not just one, so keys cannot pile up in a buffer and
        arrive seconds late.
        """
        if self._fd is None:
            return ""
        keys = []
        while select.select([self._fd], [], [], 0)[0]:
            char = os.read(self._fd, 1)
            if not char:
                break
            keys.append(char.decode(errors="ignore"))
        return "".join(keys)

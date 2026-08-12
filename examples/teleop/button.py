import os
import select
import sys
import termios
import tty
import numpy as np


class Button:


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
            # Not a real terminal (piped stdin, nohup, some IDE consoles).
            # Degrade to "no key is ever pressed" instead of killing the run.
            self._fd = None
            self._old_attrs = None
            print("[keys] stdin is not a terminal - keyboard control disabled")
        return self

    def __enter__(self):
        return self.start()
#*exc_info
    def __exit__(self ):
        self.restore()
        return False  # never swallow exceptions

    def restore(self):
        if self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def get_keys(self):
        """All keypresses waiting since the last call, as a string ('' if none).

        Returns every pending character rather than one, so a fast typist (or a
        slow control loop) cannot leave keys stuck in the buffer for seconds.
        """
        if self._fd is None:
            return ""
        keys = []
        while select.select([self._fd], [], [], 0)[0]:
            char = os.read(self._fd, 1)
            if not char:
                break  # EOF
            keys.append(char.decode(errors="ignore"))
        return "".join(keys)

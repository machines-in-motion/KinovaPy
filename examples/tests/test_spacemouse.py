import numpy as np
import time
import time
import numpy as np
from KinovaPy.utils.joy import SpaceMouseExpert
spacemouse0 = SpaceMouseExpert()


def my_app() -> None:
    while True:
        space_action, space_buttons = spacemouse0.get_action()
        
        analog_cmd = space_action.tolist()
        digital_cmd = [space_buttons[0], space_buttons[1], 0, 0]
        print(f'Digital: {digital_cmd}')
        print(f'Analog: {analog_cmd}')
        time.sleep(0.02)

if __name__ == "__main__":
    my_app()
    
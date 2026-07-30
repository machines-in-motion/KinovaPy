import numpy as np
import time
import sys
import os
from locompc.utils import load_yaml_file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KinovaPy.gim4305 import Gim4305
from KinovaPy.utils.joy import SpaceMouseExpert
spacemouse0 = SpaceMouseExpert()
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")

config_dir = os.path.join(CONFIG_PATH, 'kinova_float.yml')
config = load_yaml_file(config_dir)

QUIT = "q" in sys.argv

def main():

    run_time = config['sim_time']
    start_time = time.perf_counter()

    motor = Gim4305()
    print("Enabling motor...")
    motor.enable()
    time.sleep(0.1)
    motor.set_zero()


    while time.perf_counter()-start_time < run_time:
        space_action, space_buttons = spacemouse0.get_action()
        
        analog_cmd = space_action.tolist()
        digital_cmd = [space_buttons[0], space_buttons[1]]
        time.sleep(0.02)
        
        jk = 0
        jl = 0.03

        direct = digital_cmd[0] #left button when wire faces away from you
        direct2 = digital_cmd[1] #right button when wire faces away from you

        if direct == 1:
            motor.command(place=0, zoom=2, jk=jk, jl=jl, umph=0.05)
        elif direct2 == 1:
            motor.command(place=0, zoom=-2, jk=jk, jl=jl, umph=-0.05)
        else:
            motor.command(place=0.0, zoom=0.0,jk=jk, jl=jl, umph=0.0)
        
    motor.close()

if __name__ == "__main__":      
    main()
    
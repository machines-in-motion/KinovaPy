import numpy as np
import os
import time

import mujoco
import mujoco.viewer
from KinovaPy import SCENE_PATH


### Simulation ###
xml_name = 'scene_kinova.xml'   # choose among the scenes in g1/assets
xml_path = os.path.join(SCENE_PATH, xml_name)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

dt = 0.01
model.opt.gravity[2] = -9.81
model.opt.timestep = dt

viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.distance = 5.0
viewer.cam.azimuth = 150
viewer.cam.elevation = -20
viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.85])

for k in range(2000):
  tic = time.time()
  # mujoco.mj_step(model, data)
  viewer.sync()

  sleeptime = dt - (time.time()-tic)
  time.sleep(sleeptime if sleeptime > 0 else 0)
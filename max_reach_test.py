import numpy as np
from scipy.integrate import solve_ivp

from puma560 import Puma560
from puma_visualizer import PumaVisualizer


robot = Puma560()
viz = PumaVisualizer(robot)


T = robot.FK(np.array([0,0,0,0,0,0]))
print("FK at zero angles:")
print(T)
print("PUMA Home pose (from M):")
print(robot.M)
viz.show_pose(np.array([0,0,0,0,0,0]),title='Test pose')


import numpy as np
import puma560 as puma
from puma_visualizer import PumaVisualizer


robot = puma.Puma560()

print(robot.M)
viz   = PumaVisualizer(robot)

viz.show_pose([0]*6)
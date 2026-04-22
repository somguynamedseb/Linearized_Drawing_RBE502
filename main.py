import numpy as np
import puma560 as puma


def clean_print(T, decimals=4):
    T_clean = np.where(np.abs(T) < 1e-10, 0.0, T)
    with np.printoptions(precision=decimals, suppress=True, floatmode='fixed'):
        print(T_clean)
        
if __name__ == "__main__":
    robot = puma.Puma560()
    theta = np.array([np.pi/2, 0, 0, 0, 0, 0]) 
    T = robot.FK(theta)
    theta_out = robot.IK(T)
    
    print("Input joint angles (radians):", theta)
    clean_print(T)
    print("IK output joint angles (radians):", theta_out)
    
    

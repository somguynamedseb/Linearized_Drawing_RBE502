"""
Inspect simulation.npz and report joint-angle range issues.

Tells you:
  - max / min for each joint
  - which joints exceed [-pi, pi]
  - whether wrapping each value to [-pi, pi] still produces a continuous
    trajectory (if not, you have branch flips, not just IK winding)
"""

import numpy as np

data = np.load("simulation.npz")
q        = data["q"]            # (n, N) actual
q_target = data["q_target"]     # (n, N) commanded

def report(name, arr):
    n, N = arr.shape
    print(f"\n=== {name}  shape={arr.shape} ===")
    print(f"  {'joint':<8}{'min':>10}{'max':>10}{'span':>10}{'over_pi?':>12}")
    for j in range(n):
        mn, mx = arr[j].min(), arr[j].max()
        flag   = "YES" if (abs(mn) > np.pi or abs(mx) > np.pi) else ""
        print(f"  q{j+1:<7}{mn:>10.3f}{mx:>10.3f}{mx-mn:>10.3f}{flag:>12}")

    wrapped = (arr + np.pi) % (2*np.pi) - np.pi   # wrap to [-pi, pi]
    jumps   = np.abs(np.diff(wrapped, axis=1))
    bad     = (jumps > np.pi).sum()               # would-be discontinuities
    print(f"  After wrap: {bad} sample-to-sample jumps > pi "
          f"(branch flips if > 0)")

report("commanded q_target", q_target)
report("actual q", q)

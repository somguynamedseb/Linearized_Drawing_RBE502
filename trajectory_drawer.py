"""
Robot arm 2D trajectory drawer.

Draw a path with the mouse (click + drag, like in Paint) inside the
annulus defined by MIN_RADIUS_M and MAX_RADIUS_M. Hit "Export Trajectory"
to dump (x, y, z, t) points to trajectory.json. Time t is in seconds,
measured from the first click.
"""

import tkinter as tk
from tkinter import messagebox
import time
import math
import json

# ---- Robot workspace parameters ----------------------------------------------
MIN_RADIUS_M = 0.15
MAX_RADIUS_M = 0.7
Z_HEIGHT    = 0.2

# ---- UI parameters -----------------------------------------------------------
CANVAS_SIZE       = 800              # pixels (square canvas)
WORLD_SIZE        = 2.0              # meters spanned across the canvas
PIXELS_PER_METER  = CANVAS_SIZE / WORLD_SIZE
CENTER            = CANVAS_SIZE / 2  # pixel coords of the robot base


def pixel_to_world(px, py):
    """Convert canvas pixel (origin top-left, y down) to world meters
    (origin at base, y up)."""
    x = (px - CENTER) / PIXELS_PER_METER
    y = -(py - CENTER) / PIXELS_PER_METER
    return x, y


class TrajectoryDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Arm Trajectory Drawer")

        self.points = []          # list of (x, y, t)
        self.start_time = None    # set on first valid click
        self.last_pixel = None    # for drawing connected line segments

        # ---- Top toolbar -----------------------------------------------------
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(top, text="Export Trajectory",
                  command=self.export).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Clear",
                  command=self.clear).pack(side=tk.LEFT, padx=2)

        self.info_label = tk.Label(top, text="")
        self.info_label.pack(side=tk.LEFT, padx=10)

        self.coord_label = tk.Label(top, text="", width=35, anchor="e")
        self.coord_label.pack(side=tk.RIGHT, padx=10)

        # ---- Canvas ----------------------------------------------------------
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", cursor="crosshair")
        self.canvas.pack()

        self.draw_workspace()
        self.update_info()

        self.canvas.bind("<Button-1>",        self.on_press)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>",          self.on_motion)

    # ---- Drawing helpers -----------------------------------------------------
    def draw_workspace(self):
        max_px = MAX_RADIUS_M * PIXELS_PER_METER
        min_px = MIN_RADIUS_M * PIXELS_PER_METER

        # Reach circles
        self.canvas.create_oval(CENTER - max_px, CENTER - max_px,
                                CENTER + max_px, CENTER + max_px,
                                outline="red", width=2)
        self.canvas.create_oval(CENTER - min_px, CENTER - min_px,
                                CENTER + min_px, CENTER + min_px,
                                outline="red", width=2)

        # Axes (light dashed)
        self.canvas.create_line(0, CENTER, CANVAS_SIZE, CENTER,
                                fill="lightgray", dash=(2, 2))
        self.canvas.create_line(CENTER, 0, CENTER, CANVAS_SIZE,
                                fill="lightgray", dash=(2, 2))

        # Base marker
        self.canvas.create_oval(CENTER - 3, CENTER - 3,
                                CENTER + 3, CENTER + 3, fill="black")

        # Radius labels
        self.canvas.create_text(CENTER + max_px - 35, CENTER - 10,
                                text=f"R={MAX_RADIUS_M} m", fill="red")
        self.canvas.create_text(CENTER + min_px + 30, CENTER - 10,
                                text=f"R={MIN_RADIUS_M} m", fill="red")

    def is_in_workspace(self, x, y):
        r = math.hypot(x, y)
        return MIN_RADIUS_M <= r <= MAX_RADIUS_M

    
    
    def add_point(self, event):
        """Record a point if it's inside the annulus, draw it on canvas."""
        MIN_DT = 0.02  # seconds between recorded points (50 Hz cap)
        x, y = pixel_to_world(event.x, event.y)
        if not self.is_in_workspace(x, y):
            self.last_pixel = None  # break the line until we re-enter
            return

        if self.start_time is None:
            self.start_time = time.time()

        t = time.time() - self.start_time

        # Throttle: skip if the previous point was recorded too recently
        if self.points and t - self.points[-1][2] < MIN_DT:
            return

        self.points.append((x, y, t))

        if self.last_pixel is not None:
            self.canvas.create_line(self.last_pixel[0], self.last_pixel[1],
                                    event.x, event.y,
                                    fill="blue", width=2, capstyle=tk.ROUND)
        else:
            self.canvas.create_oval(event.x - 2, event.y - 2,
                                    event.x + 2, event.y + 2,
                                    fill="blue", outline="")
        self.last_pixel = (event.x, event.y)
        self.update_info()

    # ---- Event handlers ------------------------------------------------------
    def on_press(self, event):
        self.last_pixel = None  # ensure each new stroke starts as a dot
        self.add_point(event)
        self.update_coord(event)

    def on_drag(self, event):
        self.add_point(event)
        self.update_coord(event)

    def on_release(self, event):
        self.last_pixel = None

    def on_motion(self, event):
        self.update_coord(event)

    def update_coord(self, event):
        x, y = pixel_to_world(event.x, event.y)
        r = math.hypot(x, y)
        ok = "OK" if self.is_in_workspace(x, y) else "out"
        self.coord_label.config(
            text=f"x={x:+.3f}  y={y:+.3f}  r={r:.3f} m  [{ok}]"
        )

    def update_info(self):
        self.info_label.config(
            text=f"Points: {len(self.points)}   |   "
                 f"Z = {Z_HEIGHT:.2f} m   |   "
                 f"R range: {MIN_RADIUS_M:.2f}-{MAX_RADIUS_M:.2f} m"
        )

    # ---- Buttons -------------------------------------------------------------
    def clear(self):
        self.points = []
        self.start_time = None
        self.last_pixel = None
        self.canvas.delete("all")
        self.draw_workspace()
        self.update_info()
        self.coord_label.config(text="")

    def export(self):
        if not self.points:
            messagebox.showinfo("No data", "No points recorded yet.")
            return

        trajectory = [
            {"x": x, "y": y, "z": Z_HEIGHT, "t": t}
            for (x, y, t) in self.points
        ]

        out_path = "trajectory.json"
        with open(out_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        print(f"\nExported {len(trajectory)} points to {out_path}")
        print("First few points:")
        for p in trajectory[:5]:
            print(f"  x={p['x']:+.4f}  y={p['y']:+.4f}  z={p['z']:+.4f}  t={p['t']:.4f}s")
        if len(trajectory) > 5:
            print(f"  ... and {len(trajectory) - 5} more")

        messagebox.showinfo(
            "Exported",
            f"Saved {len(trajectory)} points to {out_path}\n"
            f"Total duration: {trajectory[-1]['t']:.2f} s"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryDrawer(root)
    root.mainloop()

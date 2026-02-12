# src/env/track.py
import math
import numpy as np


class Track:
    """
    Grid track:
      grid[y, x] == 0 -> road
      grid[y, x] == 1 -> wall
    """

    def __init__(self):
        # Match your observed mapping: mouse 716 -> grid 89 => cell_size ~ 8
        self.cell_size = 8

        # Bigger arena makes a more interesting course
        self.grid_w = 140
        self.grid_h = 90
        self.world_w = self.grid_w * self.cell_size
        self.world_h = self.grid_h * self.cell_size


        self.waypoints = [
            (120, 420),
            (220, 420),
            (320, 400),
            (420, 340),
            (520, 280),
            (620, 260),
            (720, 300),
            (760, 380),
            (720, 470),
            (620, 520),
            (500, 540),
            (380, 520),
            (260, 480),
            (160, 450),
        ]
        # Build a curvy circuit from control points (normalized 0..1)
        # This is intentionally NOT an oval: it has S-bends and tighter corners.
        ctrl = [
            (0.18, 0.78),
            (0.10, 0.62),
            (0.22, 0.48),
            (0.14, 0.30),
            (0.32, 0.22),
            (0.52, 0.28),
            (0.72, 0.20),
            (0.86, 0.30),
            (0.72, 0.40),
            (0.86, 0.52),
            (0.80, 0.72),
            (0.58, 0.78),
            (0.38, 0.70),
            (0.26, 0.82),
        ]

        # Make it a closed loop (Catmull-Rom expects wrap)
        centerline = self._catmull_rom_closed(ctrl, samples_per_seg=35)

        # Road width (in cells). Tighten this to make it harder.
        road_half_width_cells = 5  # 5 cells * 8px = 40px half-width (80px total)

        self.grid = self._rasterize_road(centerline, road_half_width_cells)

        # Start pose: near first point, facing along the tangent
        sx, sy = centerline[0]
        tx, ty = centerline[5]  # small step forward along centerline
        self.start_x = sx
        self.start_y = sy
        self.start_theta = math.atan2(ty - sy, tx - sx)

    # -------------------------
    # Public helper used by env
    # -------------------------
    def is_wall_world(self, x: float, y: float) -> bool:
        gx = int(x // self.cell_size)
        gy = int(y // self.cell_size)
        if gx < 0 or gy < 0 or gx >= self.grid_w or gy >= self.grid_h:
            return True
        return self.grid[gy, gx] == 1

    # -------------------------
    # Track generation
    # -------------------------
    def _catmull_rom_closed(self, ctrl_norm, samples_per_seg=30):
        """
        Closed Catmull-Rom spline through normalized control points.
        Returns list of points in WORLD coordinates.
        """
        # Convert to world coords (and keep away from edges a little)
        margin = 0.08
        pts = []
        for (u, v) in ctrl_norm:
            u = margin + (1 - 2 * margin) * u
            v = margin + (1 - 2 * margin) * v
            pts.append((u * self.world_w, v * self.world_h))

        n = len(pts)
        out = []
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i % n]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]

            for j in range(samples_per_seg):
                t = j / float(samples_per_seg)
                out.append(self._catmull_rom_point(p0, p1, p2, p3, t))

        return out

    def _catmull_rom_point(self, p0, p1, p2, p3, t):
        """Standard Catmull-Rom (uniform)"""
        t2 = t * t
        t3 = t2 * t

        x = 0.5 * (
            (2 * p1[0])
            + (-p0[0] + p2[0]) * t
            + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
        )
        y = 0.5 * (
            (2 * p1[1])
            + (-p0[1] + p2[1]) * t
            + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
        )
        return (x, y)

    def _rasterize_road(self, centerline_world, half_width_cells):
        """
        For each grid cell center, compute distance to nearest centerline segment.
        Mark as road if within road width, else wall.
        """
        grid = np.ones((self.grid_h, self.grid_w), dtype=np.uint8)

        # Precompute segments
        pts = centerline_world
        segs = []
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            segs.append((a, b))

        half_width_px = half_width_cells * self.cell_size

        # Fill road
        for gy in range(self.grid_h):
            cy = (gy + 0.5) * self.cell_size
            for gx in range(self.grid_w):
                cx = (gx + 0.5) * self.cell_size

                dmin = 1e9
                for (a, b) in segs:
                    d = self._dist_point_to_segment(cx, cy, a[0], a[1], b[0], b[1])
                    if d < dmin:
                        dmin = d

                if dmin <= half_width_px:
                    grid[gy, gx] = 0  # road

        # Hard boundary walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        return grid

    def _dist_point_to_segment(self, px, py, x1, y1, x2, y2):
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1

        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - x1, py - y1)

        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - x2, py - y2)

        t = c1 / c2
        projx = x1 + t * vx
        projy = y1 + t * vy
        return math.hypot(px - projx, py - projy)
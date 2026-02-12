import time
import pygame
from .track import Track
from .car import Car
from .utils import raycast
import numpy as np


class RaceTrackEnv:
    def __init__(self):
        self.track = Track()
        self.car = Car(self.track.start_x, self.track.start_y, self.track.start_theta)

        # Simulation / UI
        self.crashed = False
        self.start_time = None
        self.lap_time = 0.0
        self.best_lap_time = None
        self.laps = 0

        self.speed_mult = 1.0
        self.paused = False

        # RL config
        self.max_steps = 3000
        self.steps = 0

        # Ray sensors
        self.num_rays = 13
        self.ray_fov = 2.2            # radians (~126 degrees)
        self.ray_max_dist = 220.0
        self.show_rays = True

        # Stuck / no-progress detection
        self.stuck_steps = 0
        self.no_progress_steps = 0
        self.stuck_speed_thresh = 5.0          # px/s
        self.max_stuck_steps = 90              # ~0.75s at 60Hz
        self.max_no_progress_steps = 220       # ~2.3s at 60Hz

        # Progress system
        self.track_center = (self.track.world_w / 2, self.track.world_h / 2)
        self.prev_progress = 0.0
        self.progress_eps = 1e-4

        # Lap detection (no checkpoints)
        self.min_lap_steps = 500          # prevents "instant laps" from jitter
        self.lap_wrap_threshold = 0.95    # must be near end before wrap counts
        self.lap_reward = 50.0

        # Lap detection (start/finish line crossing)
        self.left_start_area = False
        self.start_line = None            # ((x1,y1),(x2,y2))
        self.start_radius = 60.0          # how far before we consider "left start"

    # -------------------------
    # Core RL API
    # -------------------------
    def reset(self):
        self.car.reset(self.track.start_x, self.track.start_y, self.track.start_theta)
        self.crashed = False

        self.start_time = time.time()
        self.lap_time = 0.0

        self.steps = 0
        self.stuck_steps = 0
        self.no_progress_steps = 0

        self.prev_progress = self._compute_progress()

        self.left_start_area = False
        self.start_line = self._make_start_line()

        return self._get_obs()

    def _make_start_line(self):
        """
        Create a start/finish line segment perpendicular to start heading.
        You may need to tweak half_len depending on your track width.
        """
        import math
        sx, sy = self.track.start_x, self.track.start_y
        th = self.track.start_theta

        # Perpendicular direction
        nx = -math.sin(th)
        ny =  math.cos(th)

        half_len = 50.0  # <-- adjust if needed (bigger = easier to hit)
        p1 = (sx - nx * half_len, sy - ny * half_len)
        p2 = (sx + nx * half_len, sy + ny * half_len)
        return (p1, p2)

    @staticmethod
    def _seg_intersect(a, b, c, d):
        """Return True if segment ab intersects segment cd."""
        def ccw(p, q, r):
            return (r[1]-p[1])*(q[0]-p[0]) > (q[1]-p[1])*(r[0]-p[0])
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))
    
    def _compute_progress(self):
        """
        Progress based on closest waypoint index (normalized 0..1).
        Works for any track shape as long as waypoints go around the loop.
        """
        wps = self.track.waypoints
        x, y = self.car.state.x, self.car.state.y

        # find closest waypoint
        d2 = [(x-wx)**2 + (y-wy)**2 for (wx, wy) in wps]
        idx = int(np.argmin(d2))

        return idx / float(len(wps))
    
    def _get_obs(self):
        s = self.car.state

        rays = []
        start_angle = s.theta - self.ray_fov / 2
        for i in range(self.num_rays):
            a = start_angle + i * (self.ray_fov / (self.num_rays - 1))
            d = raycast(self.track, s.x, s.y, a, self.ray_max_dist)
            rays.append(d / self.ray_max_dist)  # normalize 0..1

        # speed normalized
        v_norm = (s.v / self.car.max_speed)
        obs = np.array(rays + [v_norm], dtype=np.float32)
        return obs

    def step(self, action: int):
        """
        action in {0..8}
        returns obs, reward, done, info
        """
        prev_progress = self.prev_progress
        prev_pos = (self.car.state.x, self.car.state.y)

        self.steps += 1

        # Map discrete actions -> (throttle, steer)
        throttle = 0.0
        steer = 0.0
        if action == 0:   steer = -1.0
        elif action == 1: steer = 1.0
        elif action == 2: throttle = 1.0
        elif action == 3: throttle = -1.0
        elif action == 4: throttle = 1.0;  steer = -1.0
        elif action == 5: throttle = 1.0;  steer = 1.0
        elif action == 6: throttle = -1.0; steer = -1.0
        elif action == 7: throttle = -1.0; steer = 1.0
        elif action == 8: pass

        # Fixed physics timestep (training stability)
        dt = 1.0 / 60.0
        self.car.step(throttle, steer, dt)
        new_pos = (self.car.state.x, self.car.state.y)

        # Lap stopwatch uses sim time (not wall-clock)
        self.lap_time += dt

        # Collision check
        self.crashed = False
        for x, y in self.car.get_corners():
            if self.track.is_wall_world(x, y):
                self.crashed = True
                break

        # Progress
        progress = self._compute_progress()
        delta = progress - prev_progress

        # Wrap-around correction for delta
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0

    
        moved = (new_pos[0]-prev_pos[0])**2 + (new_pos[1]-prev_pos[1])**2
        if moved < 1.0:   # tweak threshold
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        # -------------------------
        # Reward shaping (clean)
        # -------------------------
        progress_scale = 10.0

        # main learning signal: reward forward progress, penalize backward
        reward = progress_scale * delta

        # small incentive to keep moving (only forward speed)
        reward += 0.0005 * max(self.car.state.v, 0.0)

        # time penalty
        reward -= 0.01

       

        # Stuck counter (low speed)
        if abs(self.car.state.v) < self.stuck_speed_thresh:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        # Reward shaping:
        # - forward progress reward
        # - time penalty
        

        # Proximity penalty (discourage wall hugging)
        obs = self._get_obs()
        min_ray = float(np.min(obs[:-1]))
        if min_ray < 0.10:
            reward -= (0.10 - min_ray) * 1.0

        done = False
        lap_complete = False

        # Crash ends episode
        if self.crashed:
            reward -= 5.0
            done = True

        # Termination rules
        if self.steps >= self.max_steps:
            done = True
        if self.no_progress_steps >= self.max_no_progress_steps:
            done = True
        if self.stuck_steps >= self.max_stuck_steps:
            done = True

        # Lap detection WITHOUT checkpoints:
        # Count a lap when we wrap from near 1.0 to near 0.0, after enough steps.
                # Lap detection: crossing the start/finish line (no checkpoints)
        sx, sy = self.track.start_x, self.track.start_y
        dx = new_pos[0] - sx
        dy = new_pos[1] - sy
        if not self.left_start_area:
            if (dx*dx + dy*dy) > (self.start_radius * self.start_radius):
                self.left_start_area = True

        if (not done) and self.left_start_area and (self.steps >= self.min_lap_steps):
            p1, p2 = self.start_line
            if self._seg_intersect(prev_pos, new_pos, p1, p2):
                lap_complete = True
                self.laps += 1
                reward += self.lap_reward
                done = True

                if (self.best_lap_time is None) or (self.lap_time < self.best_lap_time):
                    self.best_lap_time = self.lap_time

        # Update stored progress
        self.prev_progress = progress

        info = {
            "crashed": self.crashed,
            "steps": self.steps,
            "lap_complete": lap_complete,
            "progress": progress,
            "min_ray": min_ray,
            "stuck_steps": self.stuck_steps,
            "no_progress_steps": self.no_progress_steps,
            "lap_time": self.lap_time,
            "best_lap_time": self.best_lap_time,
            "laps": self.laps,
        }
        return obs, reward, done, info

    # -------------------------
    # Rendering / UI
    # -------------------------
    def init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.track.world_w, self.track.world_h))
        pygame.display.set_caption("DQN Race Track Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_1:
                    self.speed_mult = 1
                if event.key == pygame.K_2:
                    self.speed_mult = 2
                if event.key == pygame.K_3:
                    self.speed_mult = 5
                if event.key == pygame.K_4:
                    self.speed_mult = 10
                if event.key == pygame.K_5:
                    self.speed_mult = 30
                if event.key == pygame.K_t:
                    self.show_rays = not self.show_rays
        return True

    def render(self):
        self.screen.fill((15, 15, 20))

        cs = self.track.cell_size
        for y in range(self.track.grid_h):
            for x in range(self.track.grid_w):
                color = (30, 80, 40) if self.track.grid[y, x] == 0 else (40, 40, 55)
                pygame.draw.rect(self.screen, color, (x * cs, y * cs, cs, cs))

        if self.show_rays:
            self._draw_rays()

        corners = self.car.get_corners()
        pygame.draw.polygon(self.screen, (240, 240, 240), corners)
        if self.crashed:
            pygame.draw.polygon(self.screen, (255, 60, 60), corners, 3)

        # Start Line
        if self.start_line is not None:
            pygame.draw.line(self.screen, (80, 200, 255), self.start_line[0], self.start_line[1], 4)

        # UI text
        best = "—" if self.best_lap_time is None else f"{self.best_lap_time:.2f}s"
        progress = self._compute_progress()
        obs_ui = self._get_obs()
        center_ray = float(obs_ui[self.num_rays // 2])

        info = [
            f"Laps: {self.laps}",
            f"Lap: {self.lap_time:.2f}s   Best: {best}",
            f"Progress: {progress:.3f}",
            f"Center ray: {center_ray:.2f}",
            f"Speed x{self.speed_mult}   (1/2/3/4/5)",
            "Space pause | R reset | T toggle rays",
        ]

        y = 10
        for line in info:
            self.screen.blit(self.font.render(line, True, (230, 230, 230)), (10, y))
            y += 20

        pygame.display.flip()

    def _draw_rays(self):
        import math
        s = self.car.state
        start_angle = s.theta - self.ray_fov / 2

        for i in range(self.num_rays):
            a = start_angle + i * (self.ray_fov / (self.num_rays - 1))
            d = raycast(self.track, s.x, s.y, a, self.ray_max_dist)
            x1 = s.x + math.cos(a) * d
            y1 = s.y + math.sin(a) * d
            pygame.draw.line(self.screen, (200, 200, 80), (s.x, s.y), (x1, y1), 2)

    def tick(self):
        # render speed multiplier only affects how often we step/render in viewer
        return self.clock.tick(60) / 1000 * self.speed_mult
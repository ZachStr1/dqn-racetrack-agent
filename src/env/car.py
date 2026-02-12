import math
from dataclasses import dataclass
from .utils import clamp, wrap_angle

@dataclass
class CarState:
    x: float
    y: float
    theta: float
    v: float

class Car:
    def __init__(self, x, y, theta):
        self.state = CarState(x, y, theta, 0.0)

        self.length = 26
        self.width = 14

        self.max_speed = 260
        self.max_reverse = -80

        self.accel = 420
        self.brake = 600
        self.friction = 180
        self.max_steer_rate = 2.6

    def reset(self, x, y, theta):
        self.state = CarState(x, y, theta, 0.0)

    def step(self, throttle, steer, dt):
        s = self.state

        if throttle > 0:
            s.v += self.accel * throttle * dt
        elif throttle < 0:
            s.v -= self.brake * (-throttle) * dt
        else:
            if s.v > 0:
                s.v = max(0, s.v - self.friction * dt)
            elif s.v < 0:
                s.v = min(0, s.v + self.friction * dt)

        s.v = clamp(s.v, self.max_reverse, self.max_speed)

        speed_factor = min(1.0, abs(s.v) / 120)
        steer_rate = self.max_steer_rate * speed_factor
        s.theta = wrap_angle(s.theta + steer * steer_rate * dt)

        s.x += math.cos(s.theta) * s.v * dt
        s.y += math.sin(s.theta) * s.v * dt

    def get_corners(self):
        s = self.state
        c = math.cos(s.theta)
        si = math.sin(s.theta)

        L, W = self.length / 2, self.width / 2
        local = [(L, W), (L, -W), (-L, -W), (-L, W)]

        corners = []
        for lx, ly in local:
            x = s.x + lx * c - ly * si
            y = s.y + lx * si + ly * c
            corners.append((x, y))
        return corners
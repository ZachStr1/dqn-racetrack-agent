import math

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def wrap_angle(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi
    return theta

def raycast(track, x0: float, y0: float, angle: float, max_dist: float, step: float = 2.0) -> float:
    """
    Cast a ray from (x0,y0) at 'angle' until it hits a wall or max_dist.
    Returns distance to wall in pixels (0..max_dist).
    """
    import math
    dx = math.cos(angle)
    dy = math.sin(angle)

    dist = 0.0
    while dist < max_dist:
        x = x0 + dx * dist
        y = y0 + dy * dist
        if track.is_wall_world(x, y):
            return dist
        dist += step
    return max_dist
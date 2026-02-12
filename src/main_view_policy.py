import os
import numpy as np
import pygame

from src.env.racetrack_env import RaceTrackEnv
from src.rl.dqn_agent import DQNAgent

def normalize_q(qvals):
    qmin = float(np.min(qvals))
    qmax = float(np.max(qvals))
    if abs(qmax - qmin) < 1e-6:
        return np.zeros_like(qvals)
    return (qvals - qmin) / (qmax - qmin)  # 0..1

def draw_q_arrows(screen, car_x, car_y, car_theta, qvals, chosen_action):
    """
    Draw 9 arrows around the car. Arrow length indicates relative Q-value.
    Highlight chosen action.
    """
    import math

    qn = normalize_q(qvals)

    # Directions (just for visualization — not physics)
    # We'll map actions to arrow directions:
    # 0 left, 1 right, 2 accel, 3 brake, 4 accel+left, 5 accel+right, 6 brake+left, 7 brake+right, 8 noop
    # "Forward" is car_theta
    forward = car_theta
    left = car_theta - math.pi / 2
    right = car_theta + math.pi / 2
    back = car_theta + math.pi

    ang_map = {
        0: left,
        1: right,
        2: forward,
        3: back,
        4: (forward + left) / 2,
        5: (forward + right) / 2,
        6: (back + left) / 2,
        7: (back + right) / 2,
        8: None,  # no-op -> draw a small circle instead
    }

    base_len = 18
    extra_len = 55

    for a in range(9):
        strength = float(qn[a])
        length = base_len + strength * extra_len

        if a == 8:
            # no-op marker
            r = int(6 + strength * 8)
            color = (120, 180, 255) if a != chosen_action else (255, 80, 80)
            pygame.draw.circle(screen, color, (int(car_x), int(car_y)), r, 2)
            continue

        ang = ang_map[a]
        dx = math.cos(ang) * length
        dy = math.sin(ang) * length

        x2 = car_x + dx
        y2 = car_y + dy

        # color: chosen = red, others = bluish
        color = (255, 80, 80) if a == chosen_action else (120, 180, 255)

        pygame.draw.line(screen, color, (car_x, car_y), (x2, y2), 3)

        # arrow head
        head_len = 10
        head_ang = 0.55
        hx1 = x2 - math.cos(ang - head_ang) * head_len
        hy1 = y2 - math.sin(ang - head_ang) * head_len
        hx2 = x2 - math.cos(ang + head_ang) * head_len
        hy2 = y2 - math.sin(ang + head_ang) * head_len
        pygame.draw.line(screen, color, (x2, y2), (hx1, hy1), 3)
        pygame.draw.line(screen, color, (x2, y2), (hx2, hy2), 3)

def main():
    # ---- CHANGE THIS PATH to your actual model file ----
    # Example: runs/20260205-093000/dqn_final.pt
    MODEL_PATH = os.environ.get("MODEL_PATH", "runs/latest/dqn_final.pt")

    env = RaceTrackEnv()
    env.init_render()
    pygame.display.set_caption("DQN Race Track Agent — Policy Viewer (Q-arrows)")

    obs = env.reset()
    step_i = 0
    obs_dim = int(obs.shape[0])
    num_actions = 9

    agent = DQNAgent(obs_dim, num_actions)
    agent.load(MODEL_PATH)

    # viewer controls
    show_q = True

    running = True
    while running:
        running = env.handle_events()

        # Extra hotkeys (viewer-only)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    show_q = not show_q
                if event.key == pygame.K_t:
                    env.show_rays = not env.show_rays

        # pick greedy action (epsilon=0)
        qvals = agent.q_values(obs)
        epsilon = 0.05  # 5% randomness just for viewing
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 9)
        else:
            action = int(np.argmax(qvals))

        obs, reward, done, info = env.step(action)
        step_i += 1
        if step_i % 30 == 0:
            print(
                f"step={step_i} action={action} reward={reward:.3f} "
                f"progress={info.get('progress', 0):.3f} "
                f"min_ray={float(np.min(obs[:-1])):.2f} "
                f"stuck={info.get('stuck_steps', 0)} noprog={info.get('no_progress_steps', 0)}"
            )
        if info.get("lap_complete"):
            print(f"✅ LAP! lap_time={info.get('lap_time'):.2f}s best={info.get('best_lap_time')}")

        # Safety: if car is basically not moving for a while, force reset
        if info.get("stuck_steps", 0) > 40:
            done = True

        # render environment (track + car + rays + UI)
        env.render()

        # overlay Q-arrows on top
        if show_q:
            s = env.car.state
            draw_q_arrows(env.screen, s.x, s.y, s.theta, qvals, action)
            pygame.display.flip()

        if done:
            obs = env.reset()

    pygame.quit()

if __name__ == "__main__":
    main()
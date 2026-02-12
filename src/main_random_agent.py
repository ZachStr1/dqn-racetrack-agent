import random
import pygame
from src.env.racetrack_env import RaceTrackEnv

def main():
    env = RaceTrackEnv()
    env.init_render()
    obs = env.reset()

    print("Initial obs shape:", obs.shape)
    print("Initial rays:", obs[:-1])
    print("Initial speed:", obs[-1])

    running = True
    step = 0

    while running:
        running = env.handle_events()

        action = random.randint(0, 8)
        obs, reward, done, info = env.step(action)

        step += 1
        if step % 20 == 0:
            print(
                f"step={step} "
                f"rays[:5]={obs[:5]} "
                f"speed={obs[-1]:.2f} "
                f"reward={reward:.3f} "
                f"crashed={info['crashed']}"
            )

        env.render()

        if done:
            print("RESET\n")
            obs = env.reset()
            step = 0

    pygame.quit()

if __name__ == "__main__":
    main()
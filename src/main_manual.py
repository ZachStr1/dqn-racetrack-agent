import pygame
from src.env.racetrack_env import RaceTrackEnv

def main():
    env = RaceTrackEnv()
    env.init_render()
    env.reset()

    running = True
    while running:
        running = env.handle_events()

        keys = pygame.key.get_pressed()
        throttle = 0
        steer = 0

        if keys[pygame.K_w]:
            throttle += 1
        if keys[pygame.K_s]:
            throttle -= 1
        if keys[pygame.K_a]:
            steer -= 1
        if keys[pygame.K_d]:
            steer += 1

        dt = env.tick()
        if not env.crashed:
            env.step(throttle, steer, dt)

        env.render()

    pygame.quit()

if __name__ == "__main__":
    main()
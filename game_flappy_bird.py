import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Initialize pygame
pygame.init()
clock = pygame.time.Clock()

# Create environment
env = gym.make("FlappyBird-v0", render_mode="human")

# Reset environment
state, info = env.reset()

done = False
truncated = False

while not (done or truncated):

    # Handle window events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Detect key press continuously
    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        action = 1      # Flap
    else:
        action = 0      # No flap

    # Perform action
    state, reward, done, truncated, info = env.step(action)

    # Render game
    env.render()

    # Limit FPS
    clock.tick(30)

env.close()
pygame.quit()
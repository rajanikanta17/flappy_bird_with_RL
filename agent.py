import argparse

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        
        with open("parameters.yaml", 'r') as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
            
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.epsilon_init = params['epsilon_init']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.replay_memory_size = params['replay_memory_size']
        self.mini_batch_size = params['mini_batch_size']
        self.reward_threshold = params['reward_threshold']
        self.network_sync_rate = params['network_sync_rate']
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{param_set}.log")  
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{param_set}.pt") 
            
    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.policy_dqn = DQN(num_states, num_actions).to(device)

        # policy network created after env so we can infer sizes

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(self.policy_dqn.state_dict())
            
            steps = 0
            
            self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)
            best_reward = float('-inf')
        else:
            self.policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            self.policy_dqn.eval()
            
        for episode in itertools.count():
            
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_reward = 0.0
            done = False

            while not done and episode_reward < self.reward_threshold:
                # select action
                if is_training and random.random() < epsilon:
                    action_tensor = torch.tensor(env.action_space.sample(), dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action_tensor = self.policy_dqn(state.unsqueeze(0)).argmax(dim=1).squeeze(0).to(dtype=torch.long)

                action_int = int(action_tensor.item())

                # Processing:
                next_state, reward, terminated, truncated, info = env.step(action_int)
                done = bool(terminated or truncated)

                reward_tensor = torch.tensor(float(reward), dtype=torch.float32).to(device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)

                if is_training:
                    memory.append((state, action_tensor, next_state_tensor, reward_tensor, done))
                    steps += 1

                state = next_state_tensor
                episode_reward += float(reward)
                
            if is_training:
                print(f"Episode {episode+1} finished with reward: {episode_reward} and epsilon: {epsilon:.4f}")
            else:
                print(f"Episode {episode+1} finished with reward: {episode_reward}")
            
            #epsilon deccay
            if is_training:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                
                if episode_reward > best_reward:
                    log_message = f"New best reward: {episode_reward} at episode {episode+1}. Saving model."
                    
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')
                        
                    torch.save(self.policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
            if is_training and len(memory) >= self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, self.policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.alpha)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                                        
                                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Name of parameter set in parameters.yaml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
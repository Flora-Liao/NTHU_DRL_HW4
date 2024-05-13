from osim.env import L2M2019Env
import gym
import numpy as np
import argparse
import copy
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)

#define a env wrapper using skills: 1.frame skip 2.stack frame
#Original env:
#dict_keys(['v_tgt_field', 'pelvis', 'r_leg', 'l_leg'])
#v_tgt_field shape: (2, 11, 11) 

state_buffer = deque(maxlen=4)
start = 1

def observation_process(obs):
        v_tgt_field = obs['v_tgt_field'].flatten()
        pelvis_data = np.array([
            obs['pelvis']['height'], 
            obs['pelvis']['pitch'], 
            obs['pelvis']['roll']] + obs['pelvis']['vel']
        ).flatten()
        r_leg_data = process_leg_data(obs['r_leg']).flatten()
        l_leg_data = process_leg_data(obs['l_leg']).flatten()
        combined_observation = np.concatenate([v_tgt_field, pelvis_data, r_leg_data, l_leg_data])
        #format = observation_my = [-4.52920451e-01 -8.76109947e-01 -1.30923400e+00 ...  4.93740421e-02
        #                          9.24706027e-01  1.25377090e-12]
        #shape = (1356,)
        #type = <class 'numpy.ndarray'>
        return combined_observation

def get_concat_obs():
        return np.concatenate(list(state_buffer), axis=0)

def process_leg_data(leg):
        leg_data = np.array([
            leg['ground_reaction_forces'] + 
            [leg['joint'][k] for k in sorted(leg['joint'].keys())] +
            [leg['d_joint'][k] for k in sorted(leg['d_joint'].keys())] +
            [leg['HAB']['f'], leg['HAB']['l'], leg['HAB']['v']] +
            [leg['HAD']['f'], leg['HAD']['l'], leg['HAD']['v']] +
            [leg['HFL']['f'], leg['HFL']['l'], leg['HFL']['v']] +
            [leg['GLU']['f'], leg['GLU']['l'], leg['GLU']['v']] +
            [leg['HAM']['f'], leg['HAM']['l'], leg['HAM']['v']] +
            [leg['RF']['f'], leg['RF']['l'], leg['RF']['v']] +
            [leg['VAS']['f'], leg['VAS']['l'], leg['VAS']['v']] +
            [leg['BFSH']['f'], leg['BFSH']['l'], leg['BFSH']['v']] +
            [leg['GAS']['f'], leg['GAS']['l'], leg['GAS']['v']] +
            [leg['SOL']['f'], leg['SOL']['l'], leg['SOL']['v']] +
            [leg['TA']['f'], leg['TA']['l'], leg['TA']['v']] 
            # Add other muscles similarly
        ])
        return leg_data
    
#For Agent design
#input state dim = 1356
#input action dim = 22    

#Actor Net
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        #self.l2 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        #x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability in training
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  # log(1-tanh^2(x) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t


#SAC Agent
class Agent():
    def __init__(self):
        state_dim = 339 #339 * 4
        action_dim = 22
        self.actor = Actor(state_dim, action_dim).to(device)
        self.start = 1
        state_buffer.clear()

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Entropy coefficient
        save_dir = './SAC_env_pth/'
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.load_models(4000)

    def act(self, observation):
        state = observation
        state = observation_process(state)
        if self.start:
             for _ in range(1):
                state_buffer.append(state)
                self.start = 0
        else:
            state_buffer.pop()
            state_buffer.append(state)
        state = get_concat_obs()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _, _ = self.actor.sample(state)
        action = action.cpu().data.numpy().flatten()
        action = np.clip(action, 0.0, None)
        return action

    def load_models(self, episode_num):
        actor_path = '110062227_hw4_data'
        
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=device))
            print(f'Models loaded from episode {episode_num}')
        else:
            print('Saved models not found for the specified episode. Please check the directory or episode number.')




env = L2M2019Env(visualize=False, difficulty=2)
agent = Agent()

total_rewards = []
num_episodes = 10

for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # Convert state into a format suitable for the network
            #state = np.float32(state)
            action = agent.act(state)
            state, reward, done, _ = env.step(action)  # Assuming the env returns these values
            episode_reward += reward
        total_rewards.append(episode_reward)
        print(f'Episode {episode + 1}: Reward: {episode_reward}')
    
average_reward = np.mean(total_rewards)
print(f'Average Reward over {num_episodes} episodes: {average_reward}')
    
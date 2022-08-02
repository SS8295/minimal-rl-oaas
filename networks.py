import os
from pickletools import optimize
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import gym
import sys

class PolicyNetwork(nn.Module):
    '''
    Pytorch policy network
    This is the nn architecture that we will use for our agent.
    '''
    def __init__(self, lr, input_dims, n_actions):

        # Call the superconstructor
        super(PolicyNetwork, self).__init__()

        # Handle checkpoint and file directories
        #self.checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Neural network architecture
        # *input_dims means we are unpacking the list from input_dims
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        # Optimizer (arbitrarily chosen as Adam)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Use GPU if available, if not, use CPU (this is for NVIDIA machines)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        '''
        Calculates which action to take, given a particular state
        Input - [32, 12, 9, 16] -> state representation
        Output - [0.1, 0.1, 0.5, 0.2, 0.1] -> Distribution over available actions
        '''
        # Pass state through fully connected layer using a ReLU activation function
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
    
        return x

class PolicyGradientAgent():
    '''
    This is the RL agent that uses as a policy the PolicyNetwork to take actions.
    '''
    def __init__(self, lr, input_dims, gamma = 0.99, n_actions = 5):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        '''
        Input - [32, 12, 9, 16] -> current state of the environment
        Ouput - action
        '''
        # Convert state representation in to a PyTorch-compatible tensor
        state = T.Tensor([observation]).to(self.policy.device)

        # Get raw probabilities and activating with softmax to reassure probability distribution
        probabilities = F.softmax(self.policy.forward(state),dim = 1)

        # Feed it to a categorical distribution library to actually select an action
        action_probs = T.distributions.Categorical(probabilities)

        # Sample this distribution to get an action
        action = action_probs.sample()

        # Get log probs to feed into loss function
        log_probs = action_probs.log_prob(action)

        # Save the log_prob to the agent's memory
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        '''
        Dereference the action from the .item() function because
        OpenAI gym does not accept tensors as inputs
        '''
        self.reward_memory.append(reward)

    def learn(self):
        '''
        This function does not require an input. 
        It calculates the gradient and does backpropagation.
        '''
        # Start with reseting the gradient of the optimizer (PyTorch naturally remembers it)
        self.policy.optimizer.zero_grad()

        # G_t = R_{t+1} + gamma * R_{t+2} + gamma * R_{t+3} + ...
        # G_t = sum from k=0 to k=T {gamma**k * R_{t+k+1}}
        G = np.zeros_like(self.reward_memory)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t,len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # Convert G[t] to a PyTorch tensor
        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g,logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

class ActorCtiticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_acitons, fc1_dims=256, fc2_dims=256):
        super(ActorCtiticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_acitons)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)

class Agent():

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
                 n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCtiticNetwork(lr, input_dims, n_actions, 
                                               fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()
        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        delta = reward + self.gamma * critic_value_ * (1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

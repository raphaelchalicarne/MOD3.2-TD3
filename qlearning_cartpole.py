# -*- coding: utf-8 -*-


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 256, 10

    def __init__(self):
        super(QNet, self).__init__()
        # VOTRE CODE
        ############
        # Définition d'un réseau avec une couche cachée (à 256 neurones par exemple)
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        # VOTRE CODE
        ############
        # Calcul de la passe avant :
        # Fonction d'activation relu pour la couche cachée
        # Fonction d'activation linéaire sur la couche de sortie

        return x

class Agent:
    def __init__(self,env):

        self.env = env
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.num_episodes = 2000

        self.n_actions = env.action_space.n
        self.episode_durations = []

        self.policy_net = QNet().to(device)
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            # VOTRE CODE
            ############
            # Calcul et renvoi de l'action fournie par le réseau
        else:
            # VOTRE CODE
            ############
            # Calcul et renvoi d'une action choisie aléatoirement

    def process_state(self,state):
        return torch.from_numpy(state).unsqueeze(0).float().to(device)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # VOTRE CODE
        ############
        # Calcul de Q(s_t,a) : Q pour l'état courant

        # VOTRE CODE
        ############
        # Calcul de Q pour l'état suivant

        # VOTRE CODE
        ############
        # Calcul de Q future attendue cumulée

        # VOTRE CODE
        ############
        # Calcul de la fonction de perte de Huber

        # VOTRE CODE
        ############
        # Optimisation du modèle

    def train_policy_model(self):

        for i_episode in range(self.num_episodes):

            state = self.env.reset()

            for t in count():

                action = self.select_action(self.process_state(state))
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                if done:
                    next_state = None

                self.memory.push(self.process_state(state), action, self.process_state(next_state) if not next_state is None else None, reward)

                state = next_state

                self.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

        self.save_model()
        print('Training completed')
        plt.show()


    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./qlearning_model")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./qlearning_model", map_location=device))

    def test(self):
        print('Testing model:')
        for i_episode in range(self.num_episodes):
            print('episode: {}'.format(i_episode))

            state = self.env.reset()

            for t in count():
                self.env.render()

                # VOTRE CODE
                ############
                # Sélection d'une action appliquée à l'environnement
                # et mise à jour de l'état

        print('Testing completed')

if __name__ == '__main__':

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    env = gym.make('CartPole-v0').unwrapped
    env.reset()

    agent = Agent(env)

    # Training phase
    agent.train_policy_model()

    #Testing phase
    agent.load_model()
    agent.test()

    env.close()

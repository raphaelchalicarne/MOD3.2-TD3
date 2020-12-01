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
from skimage.transform import resize
from skimage.color import rgb2gray

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

class DQN(nn.Module):
    def __init__(self, num_frames, h, w, num_outputs):
        super(DQN, self).__init__()
        # VOTRE CODE
        ############
        # Définition du réseau. Exemple :
        # 3 couches de convolution chacune suivie d'une batch normalization
        # filtres de taille 5 pixels, pas de 2
        # 16 filtres pour la première couche
        # 32 filtres pour la deuxième
        # 64 pour la troisième
        # Finir par une couche fully connected

    def forward(self, x):
        # VOTRE CODE
        ############
        # Calcul de la passe avant :
        # Fonction d'activation relu pour les couches cachées
        # Fonction d'activation linéaire sur la couche de sortie

class Agent:
    def __init__(self,env):

        self.env = env
        self.batch_size = 32
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.num_episodes = 2000
        self.num_frames = 4

        self.im_height = 84
        self.im_width = 84

        self.n_actions = env.action_space.n
        self.actions_meaning = env.get_action_meanings()

        self.episode_durations = []

        self.policy_net = DQN(self.num_frames, self.im_height, self.im_width, self.n_actions).to(device)
        self.target_net = DQN(self.num_frames, self.im_height, self.im_width, self.n_actions).to(device)
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

    def process(self, state):
        state = resize(rgb2gray(state), (self.im_height, self.im_width), mode='reflect') * 255
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=device, dtype=torch.float)

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
            state = self.process(state)

            for t in count():

                while state.size()[1] < self.num_frames:
                        action = 1 # Fire

                        new_frame, reward, done, _ = env.step(action)
                        new_frame = self.process(new_frame)

                        state = torch.cat([state, new_frame], 1)

                action = self.select_action(state)
                new_frame, reward, done, _ = self.env.step(action.item())
                new_frame = self.process(new_frame)

                if done:
                    new_state = None
                else :
                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]

                reward = torch.tensor([reward], device=device, dtype=torch.float)

                self.memory.push(state, action, new_state, reward)

                state = new_state

                self.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

            if i_episode % self.target_update == 0:
                # Tous les targets nets sont updatés à ce moment là.
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

        self.save_model()
        print('Training completed')
        plt.show()


    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./dqn_model")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./dqn_model", map_location=device))

    def test(self):
        print('Testing model:')
        for i_episode in range(self.num_episodes):
            print('episode: {}'.format(i_episode))

            state = self.env.reset()
            state = self.process(state)

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

    env = gym.make('BreakoutDeterministic-v4').unwrapped
    env.reset()

    agent = Agent(env)

    # Training phase
    agent.train_policy_model()

    #Testing phase
    agent.load_model()
    agent.test()

    env.close()

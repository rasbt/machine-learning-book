# coding: utf-8

# Python Machine Learning, PyTorch Edition by Sebastian Raschka (https://sebastianraschka.com), Yuxi (Hayden) Liu
# (https://www.mlexample.com/) & Vahid Mirjalili (http://vahidmirjalili.com), Packt Publishing Ltd. 2021
#
# Code Repository:
#
# Code License: MIT License (https://github.com/   /LICENSE.txt)

#################################################################################
# Chapter 19 - Reinforcement Learning for Decision Making in Complex Environments
#################################################################################

# Script: agent.py

from collections import defaultdict
import numpy as np


class Agent:
    def __init__(
            self, env,
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon_greedy=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.95):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Define the q_table
        self.q_table = defaultdict(lambda: np.zeros(self.env.nA))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.env.nA)
        else:
            q_vals = self.q_table[state]
            perm_actions = np.random.permutation(self.env.nA)
            q_vals = [q_vals[a] for a in perm_actions]
            perm_q_argmax = np.argmax(q_vals)
            action = perm_actions[perm_q_argmax]
        return action

    def _learn(self, transition):
        s, a, r, next_s, done = transition
        q_val = self.q_table[s][a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma*np.max(self.q_table[next_s])

        # Update the q_table
        self.q_table[s][a] += self.lr * (q_target - q_val)

        # Adjust the epsilon
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

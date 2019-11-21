"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

import copy
from itertools import combinations
from sympy import binomial
import pandas as pd



import math

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain = {'G': [3, 3], 'B': [0.5, 0.5]}

max_time_slots = 2
channelmatrix = [[]]
n_UEs = 3
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))
scalar_gain_array = []

n_actions = binomial(n_UEs, 2)
corr_array = []

ues_thr_optimal_global = []
diff = []


class UserScheduling(object):
    def __init__(self):
        super(UserScheduling, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_UEs * 3


    def create_channel(self, channels_gain, channels_corr):
        self.mapstatetovectors(str(channels_gain)+str("_")+str(channels_corr))
        return np.concatenate((np.asarray(scalar_gain_array), np.asarray(corr_array)), axis=None)

    def reset(self, channel_state):
        global ues_thr_optimal_global
        array = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        #observations = np.array([array, channel_state], dtype=object)
        observations = np.concatenate((array, channel_state), axis= None)
        return observations

        #return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action, observation, corr_chain, state, timer_tti, channel_chain, option):
        global diff
        global ues_thr_optimal_global
        ues_thr_rl = copy.deepcopy(observation[:3])

        optimal_action, R, tmp_thr_optimal = self.get_rates(observation, action, option)
        if option == 'test':
            thr_rl = R[action]
        else:
            thr_rl = R[0]

        ues_thr_rl[action_to_ues_tbl[action][0]] += thr_rl[0]
        ues_thr_rl[action_to_ues_tbl[action][1]] += thr_rl[1]

        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))

        if timer_tti == max_time_slots:
            done = True
            if option == 'test':
                reward_optimal = 0
                for i in range(0, len(tmp_thr_optimal)):
                    reward_optimal = reward_optimal + float(np.log2(tmp_thr_optimal[i]))

                if (reward > reward_optimal):
                    diff.append(reward - reward_optimal)



        else:
            done = False

        ues_thr_optimal_global = tmp_thr_optimal
        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state, corr_chain.next_state(0))
        s_ = np.concatenate((ues_thr_rl, channels), axis= None)
        # s_ = np.array([ues_thr_rl, 'B B G_GB'], dtype=object)

        return s_, reward, done, next_channel_state



    def mapstatetovectors(self, state):
        global scalar_gain_array
        scalar_gain_array = []

        #state = state[1]
        corr = state.split("_")[1]

        global channelmatrix
        global corr_array
        ind = np.random.choice([0, 1])
        ind_1 = ind ^ 1
        ind_2 = 2
        if corr != 'BB':
            if corr == 'GB':
                corr_array = [90, 45, 45]
                channel_matrix = channel_vectors[[ind, ind_1, ind_2], :]
            else:
                corr_array = [45, 90, 45]
                channel_matrix = channel_vectors[[ind, ind_2, ind_1], :]
        else:
            corr_array = [45, 45, 90]
            channel_matrix = channel_vectors[[ind_2, ind, ind_1], :]

        channelmatrix = channel_matrix

        gain_array = state.split("_")[0].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))


    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array
        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            optimal_action, rates_per_algo, tmp_thr_optimal = self.find_optimal_action(observation, actions, option)
        else:
            optimal_action, rates_per_algo, tmp_thr_optimal = self.find_optimal_action(observation, actions, option)
        return optimal_action, rates_per_algo, tmp_thr_optimal

    def find_optimal_action(self, observation, actions_array, option):
        max_sum_log = 0
        max_ri_ti = 0
        rates = []
        global gain
        global ues_thr_optimal_global
        #global ues_thr_ri_ti_global
        tmp_max_ues_thr = []
        tmp_max_ri_ti_thr = []

        max_action = 0
        max_ri_ti_action = 0
        for action in actions_array:
            UE_1 = action_to_ues_tbl[action][0]
            UE_2 = action_to_ues_tbl[action][1]
            scalar_gain = [scalar_gain_array[UE_1], scalar_gain_array[UE_2]]
            channelmatrix_users = copy.deepcopy(channelmatrix)
            channelmatrix_users = channelmatrix_users[[UE_1, UE_2], :]
            h_inv = np.linalg.pinv(channelmatrix_users)
            h_inv_tra = np.transpose(h_inv)
            # Normalizing the inverse channel matrix
            h_inv_tra[0] = h_inv_tra[0] / np.sqrt(np.sum((np.power(h_inv_tra[0], 2))))
            h_inv_tra[1] = h_inv_tra[1] / np.sqrt(np.sum((np.power(h_inv_tra[1], 2))))
            S = []
            N = []
            sum = 0
            SINR = []
            R = []

            for i in range(0, len(channelmatrix_users)):
                channelmatrix_users[i, :] = channelmatrix_users[i, :] * scalar_gain[i]
                S.append(np.linalg.norm(np.dot(channelmatrix_users[i, :], h_inv_tra[i])))
            for i in range(0, len(channelmatrix_users)):
                array = list(range(0, len(channelmatrix_users)))
                array.remove(i)
                for j in array:
                    if np.linalg.norm(np.dot(channelmatrix_users[i, :], h_inv_tra[j])) < 10 ** -10:
                        sum = sum + 0
                    else:
                        sum = sum + np.linalg.norm(np.dot(channelmatrix_users[i, :], h_inv_tra[j]))
                N.append(sum)
                sum = 0

            for i in range(0, len(channelmatrix_users)):
                SINR.append(S[i] / (1 + N[i]))
                R.append(math.log((1 + SINR[i]), 2))
            rates.append(R)
            if option == 'test':
                ues_thr = copy.deepcopy(ues_thr_optimal_global)


                ues_thr[action_to_ues_tbl[action][0]] += rates[action][0]
                ues_thr[action_to_ues_tbl[action][1]] += rates[action][1]

                sum_log = 0
                for i in range(0, len(ues_thr)):
                    sum_log = sum_log + float(np.log2(ues_thr[i]))
                if max_sum_log <= sum_log:
                    max_action = action
                    max_sum_log = sum_log
                    tmp_max_ues_thr = copy.deepcopy(ues_thr)



        return  max_action, rates, tmp_max_ues_thr


    def render(self):
        # time.sleep(0.01)
        self.update()
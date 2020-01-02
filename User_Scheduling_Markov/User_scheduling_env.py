"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys
import pandas as pd
from matplotlib.pyplot import hist
from sympy import binomial
import math
import copy
import csv

from scipy.stats.stats import pearsonr
from MarkovChain import MarkovChain

from itertools import combinations


max_time_slots = 10
UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 3
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

algorithm = ['rl', 'random', 'optimal', 'ri/ti']

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain = {'G': [3, 3], 'B': [0.5, 0.5]}

channelmatrix = [[]]

n_actions = binomial(n_UEs, 2)

logthr_rl = []
logthr_random = []
logthr_optimal = []


ues_thr_random_global = []
ues_thr_optimal_global = []
ues_thr_ri_ti_global = []

scalar_gain_array = []

check_1 = []
check_2 = []

best_action = 0

old_optimal_action = []
old_action = []
time_window = 10
time_window_test = 2
diff = []



class UserScheduling(object):
    def __init__(self):
        super(UserScheduling, self).__init__()
        #self.file_rl = open("rl.txt", "w")
        #self.file_random = open("random.txt", "w")


        # self.action_space = ['u', 'd', 'l', 'r']
        # self.n_actions = pow(2, n_UEs)
        self.n_actions = n_actions
        # self.title('User_scheduling')
        # self.observations = np.ones((n_UEs,), dtype=int)
        # self._build_maze()
    def create_channel(self, channels_gain, channels_corr):
        return str(channels_gain)+str("_")+str(channels_corr)



    def reset(self, channel_state):
        array = np.ones((n_UEs,), dtype=float)
        observations = self.quantize(array, channel_state)
        state = np.array([array, channel_state], dtype=object)
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)
        return observations, state

    def quantize(self, array, channel_state):
        array_quantized_max = np.max(array)
        array_quantized_diff = array - array_quantized_max
        bins = pd.cut(array_quantized_diff, bins=3, labels=['0', '1', '2'], retbins=True)
        bins=bins[0]
        observations = np.array([str(bins[0] + bins[1] + bins[2]), channel_state], dtype=object)
        return observations




    def init_for_test(self):
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)

    def mapstatetovectors(self, state):

        state = state[1]
        corr = state.split("_")[1]

        global channelmatrix
        ind = np.random.choice([0, 1])
        ind_1 = ind ^ 1
        ind_2 = 2
        if corr != 'BB':
            if corr == 'GB':
                channel_matrix = channel_vectors[[ind, ind_1, ind_2], :]
            else:
                channel_matrix = channel_vectors[[ind, ind_2, ind_1], :]
        else:
            channel_matrix = channel_vectors[[ind_2, ind, ind_1], :]

        channelmatrix = channel_matrix

        gain_array = state.split("_")[0].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))

    def step(self, action, observation, corr_chain, state, timer_tti, channel_chain, episode, state_orig):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global old_action
        global old_optimal_action
        check = []

        s_ = copy.deepcopy(state_orig)
        R, tmp_thr_optimal = self.get_rates(state_orig, action, 'train')
        #old_optimal_action.append(optimal_action)
        #old_action.append(action)

        ues_thr_rl = copy.deepcopy(s_[0])

        thr_rl = R[0]

        array = list(np.arange(n_UEs))
        array.remove(action_to_ues_tbl[action][0])
        array.remove(action_to_ues_tbl[action][1])

        ues_thr_rl[array] = (1 - (1 / time_window)) * ues_thr_rl[array]
        ues_thr_rl[action_to_ues_tbl[action][0]] = (1 - (1 / time_window)) * ues_thr_rl[action_to_ues_tbl[action][0]] + (1 / time_window) * thr_rl[0]
        ues_thr_rl[action_to_ues_tbl[action][1]] = (1 - (1 / time_window)) * ues_thr_rl[action_to_ues_tbl[action][1]] + (1 / time_window) * thr_rl[1]

        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))



        if timer_tti == max_time_slots:
            done = True

        else:
            done = False

        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state, corr_chain.next_state(0))
        s_ = self.quantize(ues_thr_rl, channels)
        state_ = np.array([ues_thr_rl, channels], dtype=object)

        return s_, reward, next_channel_state, done, state_

    def step_test(self, action, observation, corr_chain, state, timer_tti, channel_chain, episode):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global old_action
        global old_optimal_action
        global ues_thr_ri_ti_global

        R, tmp_thr_optimal = self.get_rates(observation, action, 'test')
        s_ = copy.deepcopy(observation)
        ues_thr_rl = copy.deepcopy(s_[0])

        thr_rl = R[action]
        array = list(np.arange(n_UEs))
        array.remove(action_to_ues_tbl[action][0])
        array.remove(action_to_ues_tbl[action][1])

        ues_thr_rl[array] = (1 - (1 / time_window)) * ues_thr_rl[array]
        ues_thr_rl[action_to_ues_tbl[action][0]] = (1 - (1 / time_window)) * ues_thr_rl[
            action_to_ues_tbl[action][0]] + (1 / time_window) * thr_rl[0]
        ues_thr_rl[action_to_ues_tbl[action][1]] = (1 - (1 / time_window)) * ues_thr_rl[
            action_to_ues_tbl[action][1]] + (1 / time_window) * thr_rl[1]

        ues_thr_ri_ti_global = tmp_thr_optimal
        if timer_tti == time_window_test:
            done = True
            reward_optimal = 0
            for i in range(0, len(tmp_thr_optimal)):
                reward_optimal = reward_optimal + float(np.log2(tmp_thr_optimal[i]))

            reward = 0
            for i in range(0, len(ues_thr_rl)):
                reward = reward + float(np.log2(ues_thr_rl[i]))

            diff.append(reward - reward_optimal)
        else:
            done = False

        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state, corr_chain.next_state(0))
        s_ = np.array([ues_thr_rl, channels], dtype=object)

        return s_, next_channel_state, done


    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array
        scalar_gain_array = []
        self.mapstatetovectors(observation)
        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            rates_per_algo, tmp_thr_optimal = self.find_optimal_action(observation, actions, option)
            return rates_per_algo, tmp_thr_optimal
        else:
            rates_per_algo, tmp_thr_optimal = self.find_optimal_action(observation, actions, option)
            return rates_per_algo, tmp_thr_optimal

    def find_optimal_action(self, observation, actions_array, option):
        max_sum_log = 0
        max_ri_ti = 0
        rates = []
        global gain
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
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
                ues_ri_ti_thr = copy.deepcopy(ues_thr_ri_ti_global)
                ues_ri_ti_0 = rates[action][0] / ues_ri_ti_thr[action_to_ues_tbl[action][0]]
                ues_ri_ti_1 = rates[action][1] / ues_ri_ti_thr[action_to_ues_tbl[action][1]]

                array = list(actions_array)
                array.remove(action_to_ues_tbl[action][0])
                array.remove(action_to_ues_tbl[action][1])

                ues_ri_ti_thr[array] = (1 - (1 / time_window)) * ues_ri_ti_thr[array]
                ues_ri_ti_thr[action_to_ues_tbl[action][0]] = (1 - (1 / time_window)) * ues_ri_ti_thr[
                    action_to_ues_tbl[action][0]] + (1 / time_window) * rates[action][0]
                ues_ri_ti_thr[action_to_ues_tbl[action][1]] = (1 - (1 / time_window)) * ues_ri_ti_thr[
                    action_to_ues_tbl[action][1]] + (1 / time_window) * rates[action][1]

                sum_ri_ti = ues_ri_ti_0 + ues_ri_ti_1
                if max_ri_ti <= sum_ri_ti:
                    max_ri_ti_action = action
                    max_ri_ti = sum_ri_ti
                    tmp_max_ri_ti_thr = copy.deepcopy(ues_ri_ti_thr)



        return  rates, tmp_max_ri_ti_thr

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = UserScheduling()
    env.after(100, update)
    env.mainloop()

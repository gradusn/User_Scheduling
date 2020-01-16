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
from sympy import binomial
import math
import copy
import csv

from scipy.stats.stats import pearsonr
from MarkovChain import MarkovChain

from itertools import combinations


max_time_slots = 2
UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 3
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

algorithm = ['rl', 'random', 'optimal', 'ri/ti']

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain = {'G': [15, 15], 'B': [5, 5]}

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
time_window = 2
time_window_test = 2
diff = []

Max_Cqi = 16
Ues_Cqi = []

SpectralEfficiencyForCqi = [0.0, 0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55]

SpectralEfficiencyForMcs = [0.15, 0.19, 0.23, 0.31, 0.38, 0.49, 0.6, 0.74, 0.88, 1.03, 1.18,
  1.33, 1.48, 1.7, 1.91, 2.16, 2.41, 2.57,
  2.73, 3.03, 3.32, 3.61, 3.9, 4.21, 4.52, 4.82, 5.12, 5.33, 5.55,
  0, 0, 0]


McsToItbsDl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18,
  19, 20, 21, 22, 23, 24, 25, 26]

TransportBlockSizeTable = [16, 24, 32, 40, 56, 72, 88, 104, 120, 136, 144, 176, 208, 224, 256, 280, 328, 336, 376, 408, 440, 488, 520, 552, 584, 616, 712]



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
    def create_channel(self, channels_gain):
        return str(channels_gain)



    def reset(self, channel_state):
        array = np.full((1,n_UEs),0.00001, dtype=float)
        observations = np.array([array, channel_state], dtype=object)
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)
        return observations

    def init_for_test(self):
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)

    def mapstatetovectors(self, state):

        gain_array = state[1].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))

    def step(self, action, observation, state, timer_tti, channel_chain, episode):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global old_action
        global old_optimal_action
        check = []

        R, tmp_thr_optimal = self.get_rates(observation, action, 'train')

        ues_thr_rl = copy.deepcopy(observation[0])
        ues_thr_rl = ues_thr_rl[:3].flatten()


        thr_rl = R[0]*1000/1000000

        array = list(np.arange(n_UEs))
        array.remove(action)
        if (timer_tti == 1):
            ues_thr_rl[action] = thr_rl
        else:
            for i in array:
                if (ues_thr_rl[i] != 0.00001):
                    ues_thr_rl[i] = (1 - (1 / time_window)) * ues_thr_rl[i]
            if (ues_thr_rl[action] == 0.00001):
                ues_thr_rl[action] = (1 / time_window) * thr_rl
            else:
                ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl

        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))

        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state)

        if timer_tti == max_time_slots:
            s_ = self.reset(channels)
            done = True

        else:
            done = False
            s_ = np.array([ues_thr_rl, channels], dtype=object)

        return s_, reward, next_channel_state,  done

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
            UE_1 = action
            MCS = self.getMcsFromCqi(scalar_gain_array[UE_1])
            iTbs = McsToItbsDl[MCS]
            rates.append(TransportBlockSizeTable[iTbs])
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

    def getMcsFromCqi(self, ue_cqi):
        spectralEfficiency = SpectralEfficiencyForCqi[ue_cqi]
        mcs = 0
        while mcs < 28 and (SpectralEfficiencyForMcs[mcs + 1] <= spectralEfficiency):
            mcs = mcs + 1

        return mcs



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

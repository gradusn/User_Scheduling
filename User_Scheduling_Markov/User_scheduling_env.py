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


max_time_slots = 10
UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 2
n_Tx = 2
comb = combinations(np.arange(n_UEs), 2)
#action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

algorithm = ['rl', 'random', 'optimal', 'ri/ti']

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain0 = {'G': [15, 15], 'B': [9, 9]}
gain1 = {'G': [13, 13], 'B': [3, 3]}
gain = [gain0, gain1]
channelmatrix = [[]]

H = []

#n_actions = binomial(n_UEs, 2)
n_actions = 2

logthr_rl = []
logthr_random = []
logthr_optimal = []

angles = []


ues_thr_random_global = []
ues_thr_optimal_global = []
ues_thr_ri_ti_global_short = []

scalar_gain_array = []

check_1 = []
check_2 = []

best_action = 0

old_optimal_action = []
old_action = []

modevalue0 = 0
modevalue1 = 0
modevalue2 = 0
gains = []

Max_Cqi = 16
Ues_Cqi = []

SpectralEfficiencyForCqi = [0.0, 0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55]

SpectralEfficiencyForMcs = [0.15, 0.19, 0.23, 0.31, 0.38, 0.49, 0.6, 0.74, 0.88, 1.03, 1.18,
  1.33, 1.48, 1.7, 1.91, 2.16, 2.41, 2.57,
  2.73, 3.03, 3.32, 3.61, 3.9, 4.21, 4.52, 4.82, 5.12, 5.33, 5.55,
  0, 0, 0]


McsToItbsDl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18,
  19, 20, 21, 22, 23, 24, 25, 26]

TransportBlockSizeTable = [1384, 1800, 2216, 2856, 3624, 4392, 5160, 6200, 6968, 7992, 8760, 9912, 11448, 12960, 14112, 15264, 16416, 18336, 19848, 21384, 22920, 25456, 27376, 28336, 30576, 31704, 36696]
time_window = 10
time_window_short = 10

class UserScheduling(object):
    def __init__(self, value0, value1, value2, gains_dict):
        super(UserScheduling, self).__init__()

        global modevalue0
        global modevalue1
        global modevalue2
        global gains

        modevalue0 = value0
        modevalue1 = value1
        modevalue2 = value2

        gains = gains_dict

        self.n_actions = n_actions
        # self.title('User_scheduling')
        # self.observations = np.ones((n_UEs,), dtype=int)
        # self._build_maze()
    def create_channel(self, channels_gain):
        return str(channels_gain)



    def reset(self, channel_state):
        array = np.full((1,n_UEs), 0.00001, dtype=float)
        observations = np.array([array, channel_state], dtype=object)
        return observations

    def init_for_test(self):
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)


    def create_rayleigh_fading(self):

        global scalar_gain_array
        gain_channel = ""
        ray_0 = np.random.rayleigh(modevalue0, 1)
        ray_1 = np.random.rayleigh(modevalue1, 1)
        ray_2 = np.random.rayleigh(modevalue2, 1)
        scalar_gain_array = [ray_0, ray_1, ray_2]

        for i in range(0, n_UEs):
            if scalar_gain_array[i] < gains[i]['L']:
                gain_channel = gain_channel + 'L'
            elif scalar_gain_array[i] < gains[i]['M']:
                gain_channel = gain_channel + 'M'
            else:
                gain_channel = gain_channel + 'H'

        return gain_channel





    def create_guass_vectors(self):
        mean = 0
        flag = 0
        standart_deviation = np.sqrt(0.5)
        global H
        H = []
        for i in range(0, n_UEs):
            vector = []
            for j in range(0, n_Tx*2):

                coeff = np.random.normal(mean, standart_deviation)
                if j%2 == 0:
                    vector.append(coeff)
                else:
                    vector.append(complex(0,coeff))

            vector = vector / np.sqrt(np.power(np.absolute(vector[0]),2)+np.power(np.absolute(vector[1]),2)+np.power(np.absolute(vector[2]),2)+np.power(np.absolute(vector[3]),2))
            H.append(vector)
        H = np.array(H)
        bins_corr = self.find_angles()
        return bins_corr
    '''
    def find_angles(self):
        global angles
        angles = []
        for i in range(0, len(action_to_ues_tbl)):
            UE_1 = action_to_ues_tbl[i][0]
            UE_2 = action_to_ues_tbl[i][1]
            channelmatrix_users = H[[UE_1, UE_2], :]
            dot_product = np.vdot(channelmatrix_users[0],channelmatrix_users[1])
            #test_dot = np.dot(channelmatrix_users[1], channelmatrix_users[0])
            real = dot_product.real
            norm_v1 = np.sqrt(np.power(np.absolute(channelmatrix_users[0][0]), 2)+np.power(np.absolute(channelmatrix_users[0][1]), 2)+np.power(np.absolute(channelmatrix_users[0][2]), 2)+np.power(np.absolute(channelmatrix_users[0][3]), 2))
            norm_v2 = np.sqrt(np.power(np.absolute(channelmatrix_users[1][0]), 2)+np.power(np.absolute(channelmatrix_users[1][1]), 2)+np.power(np.absolute(channelmatrix_users[1][2]), 2)+np.power(np.absolute(channelmatrix_users[1][3]), 2))

            angle = np.arccos(real / (norm_v1*norm_v2))
            angles.append(np.degrees(angle))
        angles = np.array(angles)
        cut_labels_6 = ['1', '2', '3', '4','5','6']
        cut_bins = [0, 30, 60, 90, 120, 150, 180]
        binned = pd.cut(angles, bins=cut_bins, labels=cut_labels_6)
        return str(binned[0]+binned[1]+binned[2])
    '''
    def gains(self, state):
        state = state[1]
        gain_array = state.split("_")[0].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))

    def mapstatetovectors(self, state):

        gain_array = state[1].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[i][gain_array[i]]))

    def step(self, action, observation, state, timer_tti, channel_chain, episode):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global old_action
        global old_optimal_action
        check = []

        s_ = copy.deepcopy(observation)
        R, tmp_thr_optimal, optimal_action = self.get_rates(observation, action, 'train')

        ues_thr_rl = copy.deepcopy(observation[0])
        ues_thr_rl = ues_thr_rl[:3].flatten()
        thr_rl = R[0]*1000/1000000

        array = list(np.arange(n_UEs))
        array.remove(action)

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
            done = True
            s_ = self.reset(channels)
        else:
            done = False
            s_ = np.array([ues_thr_rl, channels], dtype=object)

        return s_, reward, next_channel_state,  done

    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array
        scalar_gain_array = []
        self.mapstatetovectors(observation)
        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            optimal_action, rates_per_algo, tmp_thr_optimal = self.find_action(observation, actions, option)
            return rates_per_algo, tmp_thr_optimal, optimal_action
        else:
            optimal_action, rates_per_algo, tmp_thr_optimal, tmp_thr_ri_ti, action_ri_ti = self.find_optimal_action(observation, actions, option)
            return optimal_action, rates_per_algo, tmp_thr_optimal, tmp_thr_ri_ti, action_ri_ti

    def find_action(self, observation, actions_array, option):
        max_sum_log = 0
        max_ri_ti_short = 0
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
                ues_ri_ti_thr_3_win = copy.deepcopy(ues_thr_ri_ti_global_short).flatten()
                array = list(copy.deepcopy(actions_array))
                array.remove(action)
                R_user = rates[action] * 1000 / 1000000
                ues_ri_ti_0_3_win = R_user / ues_ri_ti_thr_3_win[action]

                for i in array:
                    if (ues_ri_ti_thr_3_win[i] != 0.00001):
                        ues_ri_ti_thr_3_win[i] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_3_win[i]
                if (ues_ri_ti_thr_3_win[action] == 0.00001):
                    ues_ri_ti_thr_3_win[action] = (1 / time_window_short) * R_user
                else:
                    ues_ri_ti_thr_3_win[action] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_3_win[action] + (
                                1 / time_window_short) * R_user

                if max_ri_ti_short < ues_ri_ti_0_3_win:
                    max_ri_ti_short = ues_ri_ti_0_3_win
                    tmp_max_ri_ti_thr_short = copy.deepcopy(ues_ri_ti_thr_3_win)

            return rates, tmp_max_ri_ti_thr, tmp_max_ri_ti_thr_short

    def getMcsFromCqi(self, ue_cqi):
        spectralEfficiency = SpectralEfficiencyForCqi[ue_cqi]
        mcs = 0
        while mcs < 28 and (SpectralEfficiencyForMcs[mcs + 1] <= spectralEfficiency):
            mcs = mcs + 1

        return mcs

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
                ues_thr = copy.deepcopy(ues_thr_optimal_global)
                ues_ri_ti_thr = copy.deepcopy(ues_thr_ri_ti_global)
                ues_ri_ti_0 = rates[action][0] / ues_ri_ti_thr[action_to_ues_tbl[action][0]]
                ues_ri_ti_1 = rates[action][1] / ues_ri_ti_thr[action_to_ues_tbl[action][1]]

                ues_ri_ti_thr[action_to_ues_tbl[action][0]] += rates[action][0]
                ues_ri_ti_thr[action_to_ues_tbl[action][1]] += rates[action][1]

                ues_thr[action_to_ues_tbl[action][0]] += rates[action][0]
                ues_thr[action_to_ues_tbl[action][1]] += rates[action][1]
                sum_ri_ti = ues_ri_ti_0 + ues_ri_ti_1
                if max_ri_ti <= sum_ri_ti:
                    max_ri_ti_action = action
                    max_ri_ti = sum_ri_ti
                    tmp_max_ri_ti_thr = copy.deepcopy(ues_ri_ti_thr)
                sum_log = 0
                for i in range(0, len(ues_thr)):
                    sum_log = sum_log + float(np.log2(ues_thr[i]))
                if max_sum_log <= sum_log:
                    max_action = action
                    max_sum_log = sum_log
                    tmp_max_ues_thr = copy.deepcopy(ues_thr)
        #ues_thr_optimal_global = tmp_max_ues_thr


        return  max_action, rates, tmp_max_ues_thr, tmp_max_ri_ti_thr, max_ri_ti_action

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

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

from MarkovChain import MarkovChain


modevalue0 = 0
modevalue1 = 0
modevalue2 = 0

channelmatrix = [[]]
n_UEs = 4
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(binomial(n_UEs, 2)))
scalar_gain_array = []

cqi = []

#n_actions = binomial(n_UEs, 2)

n_actions = n_UEs

corr_array = []

ues_thr_optimal_global = []
diff = []
gains = []
n_Tx = 2

Max_Cqi = 16
Ues_Cqi = []

SpectralEfficiencyForCqi = [0.0, 0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55]

SpectralEfficiencyForMcs = [0.15, 0.19, 0.23, 0.31, 0.38, 0.49, 0.6, 0.74, 0.88, 1.03, 1.18,
  1.33, 1.48, 1.7, 1.91, 2.16, 2.41, 2.57,
  2.73, 3.03, 3.32, 3.61, 3.9, 4.21, 4.52, 4.82, 5.12, 5.33, 5.55,
  0, 0, 0]


McsToItbsDl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18,
  19, 20, 21, 22, 23, 24, 25, 26]

TransportBlockSizeTable  =  [16, 24, 32, 40, 56, 72, 88, 104, 120, 136, 144, 176, 208, 224, 256, 280, 328, 336, 376, 408, 440, 488, 520, 552, 584, 616, 712]

time_window = 10

class UserScheduling(object):
    def __init__(self, value0, value1, value2, bins):
        super(UserScheduling, self).__init__()
        global modevalue0
        global modevalue1
        global modevalue2
        global cqi
        modevalue0 = value0
        modevalue1 = value1
        modevalue2 = value2

        cqi = bins
        self.n_actions = n_actions
        #self.n_features = n_UEs * 3
        self.n_features = n_UEs * 2
        self.n_UEs = n_UEs


    def create_channel(self, channels_gain, channels_corr):
        return np.concatenate((np.asarray(channels_gain), np.asarray(channels_corr)), axis=None)

    def reset(self, channel_state):
        array = np.ones((n_UEs,), dtype=float)
        observations = np.concatenate((array, channel_state), axis= None)
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)
        return observations

    def concatenate(self, channel_state, array):
        observations = np.concatenate((array, channel_state), axis=None)
        return observations


    def init_for_test(self):
        global ues_thr_random_global
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)
        ues_thr_ri_ti_global = np.ones((n_UEs,), dtype=float)

    def create_rayleigh_fading(self, initial_channels):
        global Ues_Cqi
        Ues_Cqi = []
        global scalar_gain_array
        scalar_gain_array = []

        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.rayleigh(initial_channels[i], 1))
            for j in range(1, Max_Cqi):
                if scalar_gain_array[i] < cqi[j]:
                    Ues_Cqi.append(j)
                    break
                elif j >= Max_Cqi-1:
                    Ues_Cqi.append(Max_Cqi-1)

        return np.asarray(Ues_Cqi)

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
                    vector.append(complex(0, coeff))

            vector = vector / np.sqrt(np.power(np.absolute(vector[0]),2)+np.power(np.absolute(vector[1]),2)+np.power(np.absolute(vector[2]),2)+np.power(np.absolute(vector[3]),2))
            H.append(vector)
        H = np.array(H)
        self.find_angles()
        return angles

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


    def gains(self, state):
        state = state[1]
        gain_array = state.split("_")[0].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))

    def step(self, action, observation, initial_channels, episode, channel_chain_array):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        global old_action
        global old_optimal_action
        global time_window
        next_channel_state = []
        next_channel_state_Cqi = []

        R, tmp_thr_optimal = self.get_rates(observation, action, 'train')
        ues_thr_rl = copy.deepcopy(observation[:n_UEs])

        thr_rl = R[0]

        array = list(np.arange(n_UEs))
        array.remove(action)
        ues_thr_rl[array] = (1 - (1 / time_window)) * ues_thr_rl[array]
        ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1/time_window)*thr_rl


        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))

        for i in range(0, n_UEs):
            next_channel_state.append(channel_chain_array[i].next_state(initial_channels[i]))
        next_channel_state_Cqi = self.create_rayleigh_fading(next_channel_state)
        s_ = np.concatenate((ues_thr_rl, next_channel_state_Cqi), axis=None)

        return s_, reward, next_channel_state

    def step_test(self, action, observation, initial_channels, episode, channel_chain_array, timer_tti):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        global old_action
        global old_optimal_action
        global time_window
        next_channel_state = []
        next_channel_state_Cqi = []

        R, tmp_thr_optimal = self.get_rates(observation, action, 'test')
        ues_thr_rl = copy.deepcopy(observation[:n_UEs])
        thr_rl = R[action]

        array = list(np.arange(n_UEs))
        array.remove(action)
        ues_thr_rl[array] = (1 - (1 / time_window)) * ues_thr_rl[array]
        ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl

        ues_thr_ri_ti_global = tmp_thr_optimal
        if timer_tti == time_window-1:
            reward_optimal = 0
            for i in range(0, len(tmp_thr_optimal)):
                reward_optimal = reward_optimal + float(np.log2(tmp_thr_optimal[i]))

            reward = 0
            for i in range(0, len(ues_thr_rl)):
                reward = reward + float(np.log2(ues_thr_rl[i]))

            diff.append(reward - reward_optimal)

        for i in range(0, n_UEs):
            next_channel_state.append(channel_chain_array[i].next_state(initial_channels[i]))
        next_channel_state_Cqi = self.create_rayleigh_fading(next_channel_state)
        s_ = np.concatenate((ues_thr_rl, next_channel_state_Cqi), axis=None)

        return s_, next_channel_state


    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array

        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            TbSize, tmp_thr_optimal = self.calc_ue_rate(observation, actions, option)
            return TbSize, tmp_thr_optimal
        else:
            TbSize, tmp_thr_optimal = self.calc_ue_rate(observation, actions, option)
            return TbSize, tmp_thr_optimal



    def calc_ue_rate(self, observation, actions_array, option):
        max_sum_log = 0
        rates = []
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        global time_window
        tmp_max_ues_thr = []
        max_action = 0
        max_ri_ti = 0
        tmp_max_ri_ti_thr = []
        for action in actions_array:
            UE_1 = action
            MCS = self.getMcsFromCqi(Ues_Cqi[UE_1])
            iTbs = McsToItbsDl[MCS]
            rates.append(TransportBlockSizeTable[iTbs])
            if option == 'test':
                ues_ri_ti_thr = copy.deepcopy(ues_thr_ri_ti_global)
                ues_ri_ti_0 = rates[action] / ues_ri_ti_thr[action]
                array = list(copy.deepcopy(actions_array))
                array.remove(action)

                ues_ri_ti_thr[array] = (1-(1/time_window))*ues_ri_ti_thr[array]
                ues_ri_ti_thr[action] = (1-(1/time_window))*ues_ri_ti_thr[action] + (1/time_window)*rates[action]

                ues_thr = copy.deepcopy(ues_thr_optimal_global)
                ues_thr[action] += rates[action]
                if max_ri_ti <= ues_ri_ti_0:
                    max_ri_ti_action = action
                    max_ri_ti = ues_ri_ti_0
                    tmp_max_ri_ti_thr = copy.deepcopy(ues_ri_ti_thr)
                sum_log = 0
                for i in range(0, len(ues_thr)):
                    sum_log = sum_log + float(np.log2(ues_thr[i]))
                if max_sum_log <= sum_log:
                    max_action = action
                    max_sum_log = sum_log
                    tmp_max_ues_thr = copy.deepcopy(ues_thr)

        return rates, tmp_max_ri_ti_thr




    def getMcsFromCqi(self, ue_cqi):
        spectralEfficiency = SpectralEfficiencyForCqi[ue_cqi]
        mcs = 0
        while mcs < 28 and (SpectralEfficiencyForMcs[mcs + 1] <= spectralEfficiency):
            mcs = mcs + 1

        return mcs



    def find_action(self, observation, actions_array, option):
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
            channelmatrix_users = copy.deepcopy(H)
            channelmatrix_users = channelmatrix_users[[UE_1, UE_2], :]
            h_inv = np.linalg.pinv(channelmatrix_users)
            h_inv_tra = np.transpose(h_inv)
            # Normalizing the inverse channel matrix

            h_inv_tra[0] = h_inv_tra[0] / np.sqrt(np.power(np.absolute(h_inv_tra[0][0]),2)+np.power(np.absolute(h_inv_tra[0][1]),2)+np.power(np.absolute(h_inv_tra[0][2]),2)+np.power(np.absolute(h_inv_tra[0][3]),2))
            h_inv_tra[1] = h_inv_tra[1] / np.sqrt(np.power(np.absolute(h_inv_tra[1][0]),2)+np.power(np.absolute(h_inv_tra[1][1]),2)+np.power(np.absolute(h_inv_tra[1][2]),2)+np.power(np.absolute(h_inv_tra[1][3]),2))
            S = []
            N = []
            sum = 0
            SINR = []
            R = []

            for i in range(0, len(channelmatrix_users)):
                channelmatrix_users[i, :] = channelmatrix_users[i, :] * scalar_gain[i]
                S.append(np.dot(h_inv_tra[i], channelmatrix_users[i, :]).real)
            for i in range(0, len(channelmatrix_users)):
                array = list(range(0, len(channelmatrix_users)))
                array.remove(i)
                for j in array:

                    if np.power(np.absolute(np.dot(h_inv_tra[j], channelmatrix_users[i, :])), 2) < 10 ** -10:
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
                #ues_ri_ti_thr = copy.deepcopy(ues_thr_ri_ti_global)
                #ues_ri_ti_0 = rates[action][0] / ues_ri_ti_thr[action_to_ues_tbl[action][0]]
                #ues_ri_ti_1 = rates[action][1] / ues_ri_ti_thr[action_to_ues_tbl[action][1]]

                #ues_ri_ti_thr[action_to_ues_tbl[action][0]] += rates[action][0]
                #ues_ri_ti_thr[action_to_ues_tbl[action][1]] += rates[action][1]

                ues_thr[action_to_ues_tbl[action][0]] += rates[action][0]
                ues_thr[action_to_ues_tbl[action][1]] += rates[action][1]
                #sum_ri_ti = ues_ri_ti_0 + ues_ri_ti_1
                #if max_ri_ti <= sum_ri_ti:
                    #max_ri_ti_action = action
                    #max_ri_ti = sum_ri_ti
                    #tmp_max_ri_ti_thr = copy.deepcopy(ues_ri_ti_thr)
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
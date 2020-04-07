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


import copy
from itertools import combinations
from sympy import binomial
import pandas as pd


import math

modevalue0 = 0
modevalue1 = 0
modevalue2 = 0

channelmatrix = [[]]
n_UEs = 2
comb = combinations(np.arange(n_UEs), 2)
#action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))
scalar_gain_array = []

cqi = []

#n_actions = binomial(n_UEs, 2)
n_actions = 2
corr_array = []

ues_thr_optimal_global = []
diff = []
gains = []
n_Tx = 2

Max_Cqi = 16
Ues_Cqi = []

SpectralEfficiencyForCqi= [0.0, 0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55]

SpectralEfficiencyForMcs = [0.15, 0.19, 0.23, 0.31, 0.38, 0.49, 0.6, 0.74, 0.88, 1.03, 1.18,
  1.33, 1.48, 1.7, 1.91, 2.16, 2.41, 2.57,
  2.73, 3.03, 3.32, 3.61, 3.9, 4.21, 4.52, 4.82, 5.12, 5.33, 5.55,
  0, 0, 0]

gain0 = {'G': [15, 15], 'B': [9, 9]}
gain1 = {'G': [13, 13], 'B': [3, 3]}
gain = [gain0, gain1]
McsToItbsDl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 18,
  19, 20, 21, 22, 23, 24, 25, 26]

TransportBlockSizeTable = [1384, 1800, 2216, 2856, 3624, 4392, 5160, 6200, 6968, 7992, 8760, 9912, 11448, 12960, 14112, 15264, 16416, 18336, 19848, 21384, 22920, 25456, 27376, 28336, 30576, 31704, 36696]
time_window = 10
time_window_short = 10
time_window_test = 10

max_time_slots = 5
max_time_slots_test = 5
metric_pf = []
metric_rl = []
mean_pf = []
mean_rl = []
counter_avg = 0
take_avg = 1000


class UserScheduling(object):
    def __init__(self):
        super(UserScheduling, self).__init__()
        self.n_actions = n_actions
        #self.n_features = n_UEs * 3
        self.n_features = n_UEs * 2
        self.time_window_test = time_window_test


    def create_channel(self, state):
        global scalar_gain_array
        scalar_gain_array = []
        gain_array = state.split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[i][gain_array[i]]))

        return np.asarray(scalar_gain_array)
    def reset(self, channel_state):
        array = np.full((1, n_UEs), 1, dtype=float)
        observations = np.concatenate((array, channel_state), axis= None)

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

    def create_rayleigh_fading(self, state):
        global Ues_Cqi
        Ues_Cqi = []
        global scalar_gain_array
        gain_array = state[1].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[i][gain_array[i]]))

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

    def step(self, action, observation, state, episode, timer_tti, channel_chain):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        global old_action
        global old_optimal_action
        global time_window

        R, tmp_thr_optimal, action_pf = self.get_rates(observation, action, 'train')

        ues_thr_rl = copy.deepcopy(observation)
        ues_thr_rl = ues_thr_rl[:n_UEs]

        thr_rl = R[0]*1000/1000000

        array = list(np.arange(n_UEs))
        array.remove(action)

        for i in array:
            if ues_thr_rl[i] != 1:
                ues_thr_rl[i] = (1 - (1 / time_window)) * ues_thr_rl[i]

        if ues_thr_rl[action] == 1:
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
        else:
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl

        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))

        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state)

        if timer_tti == max_time_slots:
            channels = self.create_channel(next_channel_state)
            s_ = self.reset(channels)
            done = True

        else:
            s_ = np.concatenate((ues_thr_rl, channels), axis=None)
            done = False

        return s_, reward, done, next_channel_state

    def step_test(self, action, observation, start_state, episode, timer_tti, channel_chain):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        global old_action
        global old_optimal_action
        global time_window
        global counter_avg

        R, tmp_thr_optimal, action_pf = self.get_rates(observation, action, 'test')
        ues_thr_rl = copy.deepcopy(observation)
        ues_thr_rl = ues_thr_rl[:n_UEs]
        thr_rl = R[action]*1000/1000000

        array = list(np.arange(n_UEs))
        array.remove(action)
        print(start_state)
        print(action)


        for i in array:
            if ues_thr_rl[i] != 1:
                ues_thr_rl[i] = (1 - (1 / time_window)) * ues_thr_rl[i]

        if ues_thr_rl[action] == 1:
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
        else:
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl

        ues_thr_ri_ti_global = tmp_thr_optimal

        next_channel_state = channel_chain.next_state(start_state)
        channels = self.create_channel(next_channel_state)

        if timer_tti == max_time_slots_test:
            reward_optimal = 0
            for i in range(0, len(tmp_thr_optimal)):
                reward_optimal = reward_optimal + float(np.log2(tmp_thr_optimal[i]))

            metric_pf.append(reward_optimal)

            reward = 0
            for i in range(0, len(ues_thr_rl)):
                reward = reward + float(np.log2(ues_thr_rl[i]))

            metric_rl.append(reward)

            s_ = self.reset(channels)
            done = True
        else:
            done = False
            s_ = np.concatenate((ues_thr_rl, channels), axis=None)


        return s_, done, next_channel_state


    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array

        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            TbSize, tmp_thr_optimal, action_pf = self.calc_ue_rate(observation, actions, option)
            return TbSize, tmp_thr_optimal, action_pf
        else:
            TbSize, tmp_thr_optimal, action_pf = self.calc_ue_rate(observation, actions, option)
            return TbSize, tmp_thr_optimal, action_pf



    def calc_ue_rate(self, observation, actions_array, option):
        max_sum_log = 0
        rates = []
        global ues_thr_optimal_global
        global ues_thr_ri_ti_global
        global time_window
        tmp_max_ues_thr = []
        max_action = 0
        max_ri_ti = 0
        max_ri_ti_action = 0
        tmp_max_ri_ti_thr = []
        for action in actions_array:
            UE_1 = action
            MCS = self.getMcsFromCqi(scalar_gain_array[UE_1])
            iTbs = McsToItbsDl[MCS]
            rates.append(TransportBlockSizeTable[iTbs])
            if option == 'test':
                ues_ri_ti_thr = copy.deepcopy(ues_thr_ri_ti_global).flatten()
                ues_ri_ti_0 = rates[action] / ues_ri_ti_thr[action]

                array = list(copy.deepcopy(actions_array))
                array.remove(action)
                R_user = rates[action] * 1000 / 1000000
                ues_ri_ti_0 = R_user / ues_ri_ti_thr[action]

                for i in array:
                    if (ues_ri_ti_thr[i] != 1):
                        ues_ri_ti_thr[i] = (1 - (1 / time_window_test)) * ues_ri_ti_thr[i]

                if ues_ri_ti_thr[action] == 1:
                    ues_ri_ti_thr[action] = (1 - (1 / time_window_test)) * ues_ri_ti_thr[action] + (
                            1 / time_window_test) * R_user
                else:
                    ues_ri_ti_thr[action] = (1 - (1 / time_window_test)) * ues_ri_ti_thr[action] + (
                                1 / time_window_test) * R_user

                if max_ri_ti < ues_ri_ti_0:
                    max_ri_ti_action = action
                    max_ri_ti = ues_ri_ti_0
                    tmp_max_ri_ti_thr = copy.deepcopy(ues_ri_ti_thr)

        return rates, tmp_max_ri_ti_thr, max_ri_ti_action




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
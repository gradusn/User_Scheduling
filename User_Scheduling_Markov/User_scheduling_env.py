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


max_time_slots = 5
UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 2
comb = combinations(np.arange(n_UEs), 2)
#action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

algorithm = ['rl', 'random', 'optimal', 'ri/ti']

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain0 = {'G': [15, 15], 'B': [9, 9]}
gain1 = {'G': [13, 13], 'B': [3, 3]}
#gain2 = {'G': [11, 11], 'B': [3, 3]}

#gain0 = {'G0': [7],'G1': [12], 'G2': [15], 'B': [5]}
#gain1 = {'G0': [14], 'G1': [16], 'G2': [17], 'G3': [19], 'G4': [27], 'B': [10]}

#gain = [gain0, gain1, gain2]
gain = [gain0, gain1]

count_GG_rl = 0
count_GG_pf = 0



channelmatrix = [[]]

#n_actions = binomial(3, 2)
n_actions = 2

logthr_rl = []
logthr_random = []
logthr_optimal = []


ues_thr_random_global = []
ues_thr_ri_ti_global_short = []
ues_thr_ri_ti_global = []
ues_thr_ri_ti_global_rr = []
UE_for_RR = 0

scalar_gain_array = []

check_1 = []
check_2 = []

best_action = 0

old_optimal_action = []
old_action = []
time_window = 10
time_window_short = 10
time_window_large = 1000
time_window_test = 20
diff = []
metric_rl = []
metric_pf = []
metric_pf_short = []
metric_rr = []

mean_rl = []
mean_pf = []


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
TransportBlockSizeTable_simple = [1384, 1800, 2216, 2856, 5000, 4392, 7000, 6200, 6968, 10000, 8760, 12000, 11448, 14000, 15000, 16000, 17000, 18336, 19000, 21384, 22920, 25456, 27376, 28336, 30576, 31704, 27000]

take_avg = 100000
counter_avg = 0

Throughputs = []
array_thr_rl = []
array_thr_pf = []


q_table = pd.DataFrame(columns=list(range(3)), dtype=np.float64)
string_channels = ""

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
    def create_channel(self, channels_gain, timer_tti):
        return str(channels_gain)+ " "+str(timer_tti)



    def reset(self, channel_state):
        global Throughputs
        global array_thr_rl
        array_thr_rl = np.full((1, n_UEs), 0, dtype=float).flatten()
        Throughputs = np.full((1, n_UEs), 1, dtype=float)
        array_slots = np.full((1, n_UEs), 0, dtype=float)
        gb_slots = np.full((1, n_UEs), -1, dtype=float). flatten()
        gb_channels = channel_state.split()
        for i in range(n_UEs):
            if gb_channels[i] == 'G':
                gb_slots[i] = 0
            else:
                gb_slots[i] = 1
        observations = np.array([array_slots, channel_state, gb_slots], dtype=object)


        global ues_thr_random_global
        global ues_thr_ri_ti_global_short
        global ues_thr_ri_ti_global
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)

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
            scalar_gain_array.append(np.random.choice(gain[i][gain_array[i]]))

    def step(self, action, observation, state, timer_tti, channel_chain, episode):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global old_action
        global old_optimal_action
        global Throughputs
        check = []

        R, tmp_thr_optimal, tmp_thr_optimal_short = self.get_rates(observation, action, 'train', timer_tti)

        print(str(observation[1])+ " " + str(action) + " " +  str(observation[0]))
        #ues_thr_rl = copy.deepcopy(observation[0])
        #ues_thr_rl = ues_thr_rl[:3].flatten()
        slots = copy.deepcopy(observation[0])
        gbslots = copy.deepcopy(observation[2]).flatten()
        gb_channels = copy.deepcopy(observation[1]).split()
        #t = observation[1]
        slots = slots.flatten()
        #gbslots = gbslots.flatten()
        ues_thr_rl = Throughputs[:3].flatten()


        thr_rl = R[0]*1000/1000000

        array = list(np.arange(n_UEs))
        array.remove(action)

        array_thr_rl[action] += thr_rl

        #print(observation[0])
        next_channel_state = channel_chain.next_state(state)

        if ((slots == [0, 1]).all() and  (gbslots == [0, 2]).all() and observation[1] == 'G B 2' and action == 0):
            stop = 1



        for i in array:
            if (ues_thr_rl[i] != 1):
                ues_thr_rl[i] = (1 - (1 / time_window)) * ues_thr_rl[i]
            slots[i] += 1



        if (ues_thr_rl[action] == 1):
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
        else:
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
        slots[action] = 0

        Throughputs = ues_thr_rl
        # reward function
        reward = 0
        for i in range(0, len(array_thr_rl)):
            if (array_thr_rl[i] == 0):
                reward = reward + 0
            else:
                reward = reward + float(np.log2(array_thr_rl[i]))




        if timer_tti == max_time_slots:
            channels = self.create_channel(next_channel_state, 1)
            s_ = self.reset(channels)
            done = True

        else:

            channels = self.create_channel(next_channel_state, timer_tti+1)
            gb_channels = channels.split()
            done = False
            for i in range(n_UEs):
                if gb_channels[i] == 'G':
                    gbslots[i] = 0
                else:
                    gbslots[i] += 1

            s_ = np.array([slots, channels, gbslots], dtype=object)

        return s_, reward, next_channel_state,  done

    def step_test(self, action, observation, state, timer_tti, channel_chain, episode):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_ri_ti_global_short
        global old_action
        global old_optimal_action
        global ues_thr_ri_ti_global
        global counter_avg
        global Throughputs
        global metric_pf_short
        global metric_rl
        global count_GG_rl
        global count_GG_pf
        global string_channels
        global ues_thr_ri_ti_global_rr
        global UE_for_RR
        global metric_rr

        gbslots = copy.deepcopy(observation[2]).flatten()
        gb_channels = copy.deepcopy(observation[1]).split()
        finish_test = 0

        R, tmp_thr_optimal, tmp_thr_optimal_short, action_pf = self.get_rates(observation, action, 'test', timer_tti)
        print(str(observation) + str(action) + str(action_pf))
        string_channels = string_channels + str(observation[0]) + " " + str(observation[1]) + "  "
        if str(observation[1]) == 'G G '+str(timer_tti):
            if action == 1:
                count_GG_rl = count_GG_rl + 1
            if action_pf == 1:
                count_GG_pf = count_GG_pf + 1

        slots = copy.deepcopy(observation[0])
        slots = slots.flatten()
        ues_thr_rl = Throughputs[:3].flatten()

        ues_ri_ti_thr_rr = copy.deepcopy(ues_thr_ri_ti_global_rr).flatten()

        thr_rl = R[action] * 1000 / 1000000
        thr_rr = R[UE_for_RR] * 1000 / 1000000

        array = list(np.arange(n_UEs))
        array.remove(action)

        array_rr = list(np.arange(n_UEs))
        array_rr.remove(UE_for_RR)

        for i in array:
            if (ues_thr_rl[i] != 1):
                ues_thr_rl[i] = (1 - (1 / time_window)) * ues_thr_rl[i]
            slots[i] += 1

        if (ues_thr_rl[action] == 1):
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
        else:
            ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
        slots[action] = 0

        Throughputs = ues_thr_rl

        for i in array_rr:
            if (ues_ri_ti_thr_rr[i] != 1):
                ues_ri_ti_thr_rr[i] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_rr[i]

        if (ues_ri_ti_thr_rr[UE_for_RR] == 1):
            ues_ri_ti_thr_rr[UE_for_RR] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_rr[UE_for_RR] + (
                        1 / time_window_short) * thr_rr
        else:
            ues_ri_ti_thr_rr[UE_for_RR] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_rr[UE_for_RR] + (
                        1 / time_window_short) * thr_rr

        ues_thr_ri_ti_global = tmp_thr_optimal
        ues_thr_ri_ti_global_short = tmp_thr_optimal_short
        ues_thr_ri_ti_global_rr = ues_ri_ti_thr_rr

        UE_for_RR = UE_for_RR + 1
        UE_for_RR = UE_for_RR % n_UEs


        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state, timer_tti+1)

        if timer_tti == max_time_slots:
            channels = self.create_channel(next_channel_state, 1)

            reward = 0
            for i in range(0, len(ues_thr_rl)):
                reward = reward + float(np.log2(ues_thr_rl[i]))

            metric_rl.append(reward)

            reward_optimal_short = 0
            for i in range(0, len(tmp_thr_optimal_short)):
                reward_optimal_short = reward_optimal_short + float(np.log2(tmp_thr_optimal_short[i]))

            metric_pf_short.append(reward_optimal_short)

            reward_rr = 0
            for i in range(0, len(ues_ri_ti_thr_rr)):
                reward_rr = reward_rr + float(np.log2(ues_ri_ti_thr_rr[i]))

            metric_rr.append(reward_rr)


            #self.check_state(string_channels, reward, reward_optimal_short)

            string_channels = ""
            s_ = self.reset(channels)
            done = True

        else:
            gb_channels = channels.split()
            done = False
            for i in range(n_UEs):
                if gb_channels[i] == 'G':
                    gbslots[i] = 0
                else:
                    gbslots[i] += 1

            s_ = np.array([slots, channels, gbslots], dtype=object)
            #s_ = np.array([slots, channels], dtype=object)

        if counter_avg == take_avg:
            counter_avg = 0
            mean_rl.append(np.mean(metric_rl))
            mean_pf.append(np.mean(metric_pf_short))
            finish_test = 1
            #metric_rl = []
            #metric_pf_short = []

        counter_avg = counter_avg + 1

        return s_, next_channel_state, done, finish_test

    def check_state(self, channels, reward_rl, reward_pf):
        global q_table
        if channels not in q_table.index:
            q_table = q_table.append(
                pd.Series(
                    [0] * 3,
                    index=q_table.columns,
                    name=channels,
                )
            )
            q_table.loc[channels, 0] += 1
            q_table.loc[channels, 1] = np.round(reward_rl, 5)
            q_table.loc[channels, 2] = np.round(reward_pf, 5)
        else:
            q_table.loc[channels, 0] += 1



    def get_rates(self, observation, action_rl, option, timer_tti):
        global scalar_gain_array
        scalar_gain_array = []
        self.mapstatetovectors(observation)
        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            rates_per_algo, tmp_thr_optimal, tmp_thr_optimal_short, action_pf = self.find_optimal_action(observation, actions, option, timer_tti)
            return rates_per_algo, tmp_thr_optimal, tmp_thr_optimal_short
        else:
            rates_per_algo, tmp_thr_optimal, tmp_thr_optimal_short, action_pf = self.find_optimal_action(observation, actions, option, timer_tti)
            return rates_per_algo, tmp_thr_optimal, tmp_thr_optimal_short, action_pf

    def find_optimal_action(self, observation, actions_array, option, timer_tti):
        max_sum_log = 0
        max_ri_ti = 0
        max_ri_ti_short = 0
        rates = []
        global gain
        global ues_thr_ri_ti_global_short
        global ues_thr_ri_ti_global
        global ues_thr_ri_ti_global_rr
        tmp_max_ri_ti_thr_short = []
        tmp_max_ri_ti_thr = []

        max_action = 0
        max_ri_ti_action = 0
        for action in actions_array:
            UE_1 = action
            MCS = self.getMcsFromCqi(scalar_gain_array[UE_1])
            iTbs = McsToItbsDl[MCS]
            rates.append(TransportBlockSizeTable[iTbs])
            #rates.append(TransportBlockSizeTable_simple[scalar_gain_array[UE_1]-1])
            if option == 'test':
                #ues_ri_ti_thr = copy.deepcopy(ues_thr_ri_ti_global).flatten()
                ues_ri_ti_thr_3_win = copy.deepcopy(ues_thr_ri_ti_global_short).flatten()


                array = list(copy.deepcopy(actions_array))
                array.remove(action)
                R_user = rates[action]*1000/1000000
                #ues_ri_ti_0 = R_user / ues_ri_ti_thr[action]
                ues_ri_ti_0_3_win = R_user / ues_ri_ti_thr_3_win[action]

                for i in array:
                    if (ues_ri_ti_thr_3_win[i] != 1):
                        ues_ri_ti_thr_3_win[i] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_3_win[i]

                if (ues_ri_ti_thr_3_win[action] == 1):
                    ues_ri_ti_thr_3_win[action] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_3_win[action] + (1 / time_window_short) * R_user
                else:
                    ues_ri_ti_thr_3_win[action] = (1 - (1 / time_window_short)) * ues_ri_ti_thr_3_win[action] + (1 / time_window_short) * R_user

                if max_ri_ti_short < ues_ri_ti_0_3_win:
                    max_ri_ti_action = action
                    max_ri_ti_short = ues_ri_ti_0_3_win
                    tmp_max_ri_ti_thr_short = copy.deepcopy(ues_ri_ti_thr_3_win)



        return  rates, tmp_max_ri_ti_thr, tmp_max_ri_ti_thr_short, max_ri_ti_action

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
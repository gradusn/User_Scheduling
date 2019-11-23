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

    def step(self, action, observation, corr_chain, state, timer_tti, channel_chain, episode, option):
        global ues_thr_random_global
        global scalar_gain_array
        global ues_thr_optimal_global
        global old_action
        global old_optimal_action
        check = []

        s_ = copy.deepcopy(observation)
        R, tmp_thr_optimal, optimal_action = self.get_rates(observation, action, option)
        #old_optimal_action.append(optimal_action)
        #old_action.append(action)

        ues_thr_rl = copy.deepcopy(s_[0])

        thr_rl = R[0]

        ues_thr_rl[action_to_ues_tbl[action][0]] += thr_rl[0]
        ues_thr_rl[action_to_ues_tbl[action][1]] += thr_rl[1]

        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))



        if timer_tti == max_time_slots:
            done = True
            '''
            optimal_sum_log = 0
            for i in range(0, len(tmp_thr_optimal)):
                optimal_sum_log = optimal_sum_log + float(np.log2(tmp_thr_optimal[i]))
            '''
            '''
            with open("reward_sum_log_3_tti.csv", "a") as reward_sum_log:
                reward_sum_log_csv = csv.writer(reward_sum_log, dialect='excel')
                reward_sum_log_csv.writerow([reward])
                reward_sum_log.close()
            with open("optimal_sum_log_3_tti.csv", "a") as optimal_log:
                optimal_sum_log_csv = csv.writer(optimal_log, dialect='excel')
                optimal_sum_log_csv.writerow([optimal_sum_log])
                optimal_log.close()

            if reward > optimal_sum_log:
                check.append([reward, optimal_sum_log, observation[1], observation_old[1], episode, best_action, old_optimal_action, old_action, state_action, state_action_old])
                with open("reward_vs_optimal_sum_log_3_tti.csv", "a") as sum_log:
                    sum_log_csv = csv.writer(sum_log, dialect='excel')
                    sum_log_csv.writerow(check)
                    sum_log.close()

                old_optimal_action = []
                old_action = []
            '''
        else:
            done = False
            #ues_thr_optimal_global = tmp_thr_optimal
        next_channel_state = channel_chain.next_state(state)
        channels = self.create_channel(next_channel_state, corr_chain.next_state(0))
        s_ = np.array([ues_thr_rl, channels], dtype=object)
        #s_ = np.array([ues_thr_rl, 'B B G_GB'], dtype=object)

        return s_, reward, next_channel_state,  done

    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array
        scalar_gain_array = []
        self.mapstatetovectors(observation)
        global gain
        actions = np.arange(n_UEs)
        if option == 'train':
            actions = [action_rl]
            optimal_action, rates_per_algo, tmp_thr_optimal = self.find_optimal_action(observation, actions, option)
            return rates_per_algo, tmp_thr_optimal, optimal_action
        else:
            optimal_action, rates_per_algo, tmp_thr_optimal, tmp_thr_ri_ti, action_ri_ti = self.find_optimal_action(observation, actions, option)
            return optimal_action, rates_per_algo, tmp_thr_optimal, tmp_thr_ri_ti, action_ri_ti

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

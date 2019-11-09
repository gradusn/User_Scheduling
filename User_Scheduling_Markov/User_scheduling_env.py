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
n_Tx = 2
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

algorithm = ['rl', 'random', 'optimal', 'ri/ti']

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain = {'G': [3, 3], 'B': [0.5, 0.5]}

channelmatrix = [[]]

H = []

n_actions = binomial(n_UEs, 2)

logthr_rl = []
logthr_random = []
logthr_optimal = []

angles = []


ues_thr_random_global = []
ues_thr_optimal_global = []
ues_thr_ri_ti_global = []

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

    def gains(self, state):
        state = state[1]
        gain_array = state.split("_")[0].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))

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
        #next_channel_state = channel_chain.next_state(state)
        next_channel_state = self.create_rayleigh_fading()
        channes_guass_corr = self.create_guass_vectors()
        channels = self.create_channel(next_channel_state, channes_guass_corr)
        s_ = np.array([ues_thr_rl, channels], dtype=object)

        return s_, reward, next_channel_state,  done

    def get_rates(self, observation, action_rl, option):
        global scalar_gain_array
        #scalar_gain_array = []
        #self.create_guass_vectors()
        #self.gains(observation)
        #self.mapstatetovectors(observation)
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

            #v1 = []

            #t = complex(1,1)
            #t2 = complex(-1,0)
            #v1 = [t, t2]
            #v3 = [t.real,complex(0,t.imag),t2.real,complex(0,t2.imag)]
            #v1 = v1 / np.sqrt(np.power(np.absolute(v1[0]),2)+np.power(np.absolute(v1[1]),2))
            #v3 = v3 / np.sqrt(np.power(np.absolute(v3[0]),2)+np.power(np.absolute(v3[1]),2)+np.power(np.absolute(v3[2]),2)+np.power(np.absolute(v3[3]),2))

            #print(v3)

            #norm = np.sqrt(np.vdot(v1[0],v1[0])+np.vdot(v1[1],v1[1]))
            #norm2 = np.sqrt(np.vdot(v3[0],v3[0])+np.vdot(v3[1],v3[1])+np.vdot(v3[2],v3[2])+np.vdot(v3[3],v3[3]))


            #c1 = complex(1,1)
            #c2 = complex(1.9,0)
            #v2 = [c1, c2]
            #v4 = [c1.real,complex(0,c1.imag),c2.real,complex(0,c2.imag)]
            #v4 = v4 / np.sqrt(np.power(np.absolute(v4[0]),2)+np.power(np.absolute(v4[1]),2)+np.power(np.absolute(v4[2]),2)+np.power(np.absolute(v4[3]),2))

            #v2 = v2 / np.sqrt(np.power(np.absolute(v2[0]),2)+np.power(np.absolute(v2[1]),2))


            #channel = [v3, v4]
            #test2 = np.linalg.pinv(channel)
            #test2 = np.transpose(test2)

            #test2[0] = test2[0] / np.sqrt(np.power(np.absolute(test2[0][0]), 2) + np.power(np.absolute(test2[0][1]), 2)+np.power(np.absolute(test2[0][2]), 2)+np.power(np.absolute(test2[0][3]), 2))
            #test2[1] = test2[1] / np.sqrt(np.power(np.absolute(test2[1][0]), 2) + np.power(np.absolute(test2[1][1]), 2)+np.power(np.absolute(test2[1][2]), 2)+np.power(np.absolute(test2[1][3]), 2))

            #bla2 = np.vdot(channel[0], test2[0])
            #bla3 = np.dot(channel[1], test2[1])


            #bla6 = np.dot(channel[0], test2[1])
            #bla7 = np.dot(channel[1], test2[0])

            #dot_product = np.vdot(channel[0], channel[1])

            #real = dot_product.real
            #abs = np.absolute(dot_product)
            #norm_v1 = np.sqrt(np.power(np.absolute(channel[0][0]), 2)+np.power(np.absolute(channel[0][1]), 2)+np.power(np.absolute(channel[0][2]), 2)+np.power(np.absolute(channel[0][3]), 2))
            #norm_v2 = np.sqrt(np.power(np.absolute(channel[1][0]), 2)+np.power(np.absolute(channel[1][1]), 2)+np.power(np.absolute(channel[1][2]), 2)+np.power(np.absolute(channel[1][3]), 2))

            #angle = np.arccos(real / (norm_v1*norm_v2))
            #angle = np.degrees(angle)

            #angle_abs = np.arccos(abs / (norm_v1*norm_v2))
            #angle_abs = np.degrees(angle_abs)

            #norm = np.sqrt(np.vdot(channelmatrix_users[0][0],channelmatrix_users[0][0])+np.vdot(channelmatrix_users[0][1],channelmatrix_users[0][1]))
            #norm = np.sqrt(np.vdot(h_inv_tra[0][0],h_inv_tra[0][0])+np.vdot(h_inv_tra[0][1],h_inv_tra[0][1]))



            for i in range(0, len(channelmatrix_users)):
                channelmatrix_users[i, :] = channelmatrix_users[i, :] * scalar_gain[i]
                S.append(np.dot(h_inv_tra[i], channelmatrix_users[i, :]).real)
            for i in range(0, len(channelmatrix_users)):
                array = list(range(0, len(channelmatrix_users)))
                array.remove(i)
                for j in array:
                    #test = np.vdot(h_inv_tra[j], channelmatrix_users[i, :])
                    #bla = h_inv_tra[j][0]*channelmatrix_users[i, :][0]+h_inv_tra[j][1]*channelmatrix_users[i, :][1]
                    #bla2 = np.conj(h_inv_tra[j][0])*channelmatrix_users[i, :][0]+np.conj(h_inv_tra[j][1])*channelmatrix_users[i, :][1]
                    #bla3 = h_inv_tra[j][0]*np.conj(channelmatrix_users[i, :][0])+h_inv_tra[j][1]*np.conj(channelmatrix_users[i, :][1])
                    #test2 = np.dot(h_inv_tra[j],channelmatrix_users[i, :] )
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


        return  max_action, rates, tmp_max_ues_thr



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

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
from MarkovChain import MarkovChain


from scipy.stats.stats import pearsonr
from MarkovChain import MarkovChain

from itertools import combinations



UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 3
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

algorithm = ['random', 'rl', 'optimal']

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

scalar_gain_array = []

check_1 = []
check_2 = []
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



    def reset(self, channel_state):
        global ues_thr_random_global
        global ues_thr_optimal_global
        #self.mapstatetovectors(channel_state)
        array = np.ones((n_UEs,), dtype=float)
        ues_thr_random_global = np.ones((n_UEs,), dtype=float)
        ues_thr_optimal_global = np.ones((n_UEs,), dtype=float)

        self.observations = np.array([array, channel_state], dtype=object)
        # print(test [1])
        # self.observations = np.ones((n_UEs,), dtype=int)
        # self.update()
        # time.sleep(0.5)
        # self.canvas.delete(self.rect)
        # origin = np.array([20, 20])
        # self.rect = self.canvas.create_rectangle(
        #   origin[0] - 15, origin[1] - 15,
        #  origin[0] + 15, origin[1] + 15,
        # fill='red')
        # return observation
        return self.observations

    def mapstatetovectors(self, state):
        #gain_array = state.split("_")[0]
        #gain_array = [gain_array.split()[0], gain_array.split()[1], gain_array.split()[2]]

        #global gain
        state = state[1]
        corr = state.split("_")[1]

        global channelmatrix
        ind = np.random.choice([0, 1])
        ind_1 = ind ^ 1
        ind_2 = 2
        if corr != 'BB':
            #ind2 = np.random.choice([2, 3])
            if corr == 'GB':
                channel_matrix = channel_vectors[[ind, ind_1, ind_2], :]
            else:
                channel_matrix = channel_vectors[[ind, ind_2, ind_1], :]
            #channelmatrix = channel_matrix
        else:
            channel_matrix = channel_vectors[[ind_2, ind, ind_1], :]

        channelmatrix = channel_matrix

        gain_array = state.split("_")[0].split()
        for i in range(0, n_UEs):
            scalar_gain_array.append(np.random.choice(gain[gain_array[i]]))



        # print(gain_1)
        # t = gain[gain_1]
        # print(gain[gain_1])
        # print(np.random.choice(gain[gain_1]))
        # scale = np.random.choice(gain[gain_1])

        # t = np.asarray(UE1)
        # print(np.asarray(UE2))

        # print(UE1.flatten())
        # channe_matrix = np.matrix(UE1.flatten(), UE2.flatten(), UE3.flatten())
        # print(channe_matrix)




    def step(self, action, observation, timer_tti, channel_chain, episode, observation_old):
        global ues_thr_random_global
        global scalar_gain_array

        self.mapstatetovectors(observation)
        # s = self.canvas.coords(self.rect)
        s_ = copy.deepcopy(observation)
        #print("The observation is: "+str(observation))
        #R, action_optimal, action_random, action_rl = self.find_optimal_Action(observation, action)
        R, action_random, sum_log_optimal, optimal_action = self.get_rates(observation, action)
        #print(s_[0])
        ues_thr_rl = copy.deepcopy(s_[0])
        ues_thr_random = ues_thr_random_global
        #ues_thr_optimal = ues_thr_optimal_global

        #print(action_to_ues_tbl[action][0])
        #print(ues_thr[action_to_ues_tbl[action][0]])

        #print("The Rate is: "+str(R))
        thr_rl = R[algorithm.index('rl')]
        thr_random = R[algorithm.index('random')]
        #thr_optimal = R[action_optimal]

        #print(ues_thr_rl)

        ues_thr_rl[action_to_ues_tbl[action][0]] += thr_rl[0]
        ues_thr_rl[action_to_ues_tbl[action][1]] += thr_rl[1]

        #ues_thr_optimal[action_to_ues_tbl[action_optimal][0]] += thr_optimal[0]
        #ues_thr_optimal[action_to_ues_tbl[action_optimal][1]] += thr_optimal[1]

        #print(ues_thr_rl)

        #print(ues_thr_random)

        ues_thr_random[action_to_ues_tbl[action_random][0]] += thr_random[0]
        ues_thr_random[action_to_ues_tbl[action_random][1]] += thr_random[1]

        #print(ues_thr_random)



        #print(s_)
        #print(observation)
        # self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        # reward function
        reward = 0
        for i in range(0, len(ues_thr_rl)):
            reward = reward + float(np.log2(ues_thr_rl[i]))

        if timer_tti == 2:
            done = True
            logthr_rl.append(reward)
            calc_thr_random = 0
            for i in range(0, len(ues_thr_random)):
                calc_thr_random = calc_thr_random + float(np.log2(ues_thr_random[i]))
            logthr_random.append(calc_thr_random)
            logthr_optimal.append(sum_log_optimal)

            if reward > sum_log_optimal:
                check = 1

            if episode == 149999:
                a = np.asarray(logthr_rl)
                np.savetxt("logthr_rl_150000.csv", a, delimiter=",")
                b = np.asarray(logthr_random)
                np.savetxt("logthr_random_150000.csv", b, delimiter=",")
                c = np.asarray(logthr_optimal)
                np.savetxt("logthr_optimal_150000.csv", c, delimiter=",")


                #self.file_rl.write(str(logthr_rl))
                #self.file_random.write(str(logthr_random))

        else:
            done = False

        s_ = np.array([ues_thr_rl, channel_chain.next_state()], dtype=object)
        scalar_gain_array = []

        return s_, reward, done


    def get_rates(self, observation, action_rl):
        rates = []
        global gain
        #self.mapstatetovectors(observation)



        for choice in algorithm:

            if choice == 'rl':
                UE_1 = action_to_ues_tbl[action_rl][0]
                UE_2 = action_to_ues_tbl[action_rl][1]
                scalar_gain = [scalar_gain_array[UE_1], scalar_gain_array[UE_2]]
            elif choice == 'random':
                action = np.random.choice(range(0, n_actions))
                UE_1 = action_to_ues_tbl[action][0]
                UE_2 = action_to_ues_tbl[action][1]
                scalar_gain = [scalar_gain_array[UE_1], scalar_gain_array[UE_2]]

            elif choice == 'optimal':
                max_optimal_sum_log, optimal_action = self.find_optimal_Action(observation)
                continue;




           # print("chosen UEs: " + str(UE_1) + "," + str(UE_2))
            #gain_array = observation[1].split("_")[0]
            #gain_array = [gain_array.split()[UE_1], gain_array.split()[UE_2]]
            #print("The gains for the UEs are: " + str(gain_array[0]) + "," + str(gain_array[1]))
            channelmatrix_users = copy.deepcopy(channelmatrix)
            channelmatrix_users = channelmatrix_users[[UE_1, UE_2], :]
            #print("The chosen UEs vectors are: " + str(channelmatrix_users[0, :]) + "," + str(channelmatrix_users[1, :]))
            h_inv = np.linalg.pinv(channelmatrix_users)
            h_inv_tra = np.transpose(h_inv)
            #Normalizing the inverse channel matrix
            h_inv_tra[0] = h_inv_tra[0] / np.sqrt(np.sum((np.power(h_inv_tra[0], 2))))
            h_inv_tra[1] = h_inv_tra[1] / np.sqrt(np.sum((np.power(h_inv_tra[1], 2))))
            #corrlation_number = np.linalg.cond(channelmatrix_users, np.inf)
            #print("The correlation between UEs is: " + str(corrlation_number))
            S = []
            N = []
            sum = 0
            SINR = []
            R = []

            for i in range(0, len(channelmatrix_users)):

                #scalar_gain = np.random.choice(gain[gain_array[i]])
                #print("The chosen scale is: " + str(scalar_gain))
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
            #print("The SINR is: " + str(SINR))
            #print(rates)

        return rates, action, max_optimal_sum_log, optimal_action

    def find_optimal_Action(self, observation):

        max_sum_log = 0
        rates = []
        global gain
        global ues_thr_optimal_global
        # self.mapstatetovectors(observation)

        # action_random = np.random.choice(range(0, n_actions))

        for action in range(0, n_actions):
            UE_1 = action_to_ues_tbl[action][0]
            UE_2 = action_to_ues_tbl[action][1]
            scalar_gain = [scalar_gain_array[UE_1], scalar_gain_array[UE_2]]

            # print("chosen UEs: " + str(UE_1) + "," + str(UE_2))
            #gain_array = observation[1].split("_")[0]
            #gain_array = [gain_array.split()[UE_1], gain_array.split()[UE_2]]
            # print("The gains for the UEs are: " + str(gain_array[0]) + "," + str(gain_array[1]))
            channelmatrix_users = copy.deepcopy(channelmatrix)
            channelmatrix_users = channelmatrix_users[[UE_1, UE_2], :]
            # print("The chosen UEs vectors are: " + str(channelmatrix_users[0, :]) + "," + str(channelmatrix_users[1, :]))
            h_inv = np.linalg.pinv(channelmatrix_users)
            h_inv_tra = np.transpose(h_inv)
            # Normalizing the inverse channel matrix
            h_inv_tra[0] = h_inv_tra[0] / np.sqrt(np.sum((np.power(h_inv_tra[0], 2))))
            h_inv_tra[1] = h_inv_tra[1] / np.sqrt(np.sum((np.power(h_inv_tra[1], 2))))
            # corrlation_number = np.linalg.cond(channelmatrix_users, np.inf)
            # print("The correlation between UEs is: " + str(corrlation_number))
            S = []
            N = []
            sum = 0
            SINR = []
            R = []

            for i in range(0, len(channelmatrix_users)):
                #scalar_gain = np.random.choice(gain[gain_array[i]])
                # print("The chosen scale is: " + str(scalar_gain))
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
        ues_thr_optimal_global = tmp_max_ues_thr

        return max_sum_log, max_action

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

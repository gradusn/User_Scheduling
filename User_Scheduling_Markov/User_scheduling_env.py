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

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 3
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))

channel_vectors = np.array(
    [[1, 0], [0, 1], [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), -1 / math.sqrt(2)]])

gain = {'G': [1, 1], 'B': [1, 1]}

channelmatrix = [[]]


class UserScheduling(object):
    def __init__(self):
        super(UserScheduling, self).__init__()
        # self.action_space = ['u', 'd', 'l', 'r']
        # self.n_actions = pow(2, n_UEs)
        self.n_actions = binomial(n_UEs, 2)
        # self.title('User_scheduling')
        # self.observations = np.ones((n_UEs,), dtype=int)
        # self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self, channel_state):
        #self.mapstatetovectors(channel_state)
        array = np.ones((n_UEs,), dtype=float)
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

    def step(self, action, observation, timer_tti, channel_chain):
        global gain
        self.mapstatetovectors(observation)
        # s = self.canvas.coords(self.rect)
        s_ = copy.deepcopy(observation)
        print("The observation is: "+str(observation))
        UE_1 = action_to_ues_tbl[action][0]
        UE_2 = action_to_ues_tbl[action][1]

        print("chosen UEs: "+str(UE_1)+","+str(UE_2))

        gain_array = s_[1].split("_")[0]
        gain_array = [gain_array.split()[UE_1], gain_array.split()[UE_2]]

        print("The gains for the UEs is: "+str(gain_array[0])+","+str(gain_array[1]))



        channelmatrix_users = channelmatrix[[UE_1, UE_2], :]
        print("The chosen UEs vectors are: "+str(channelmatrix_users[0, :])+","+str(channelmatrix_users[1, :]))
        test2 = np.transpose(channelmatrix_users)
        t = np.linalg.pinv(channelmatrix_users)
        #t_1 = np.linalg.pinv(test2)

        t_tra = np.transpose(t)
        #t_tra_2 = np.transpose(t_1)

        t_tra[0] = t_tra[0] / np.sqrt(np.sum((np.power(t_tra[0], 2))))
        t_tra[1] = t_tra[1] / np.sqrt(np.sum((np.power(t_tra[1], 2))))
        o = np.matmul(channelmatrix_users[0], t_tra[0])
        o_1 = np.matmul(channelmatrix_users[1], t_tra[0])
        o_2 = np.matmul(channelmatrix_users[1], t_tra[1])
        o_3 = np.matmul(channelmatrix_users, t)
        o_4 = np.linalg.cond(channelmatrix_users, np.inf)
        print("The correlation between UEs is: "+str(o_4))
        S = []
        N = []
        sum = 0
        SINR = []
        R = []
        for i in range(0, len(channelmatrix_users)):
            scalar_gain = np.random.choice(gain[gain_array[i]])
            print("The chosen scale is: "+str(scalar_gain))
            channelmatrix_users[i, :] = channelmatrix_users[i, :] * scalar_gain
            tmp = np.dot(channelmatrix_users[i, :], t_tra[i])
            S.append(np.linalg.norm(np.dot(channelmatrix_users[i, :], t_tra[i])))
            #print (S)
        for i in range(0, len(channelmatrix_users)):
            array = list(range(0, len(channelmatrix_users)))
            array.remove(i)
            for j in array:
                if np.linalg.norm(np.dot(channelmatrix_users[i, :], t_tra[j])) < 10 **-10:
                    sum = sum + 0
                else:
                    sum = sum + np.linalg.norm(np.dot(channelmatrix_users[i, :], t_tra[j]))
            N.append(sum)
            sum = 0

        for i in range(0, len(channelmatrix_users)):
            SINR.append(S[i]/(1+N[i]))
            R.append(math.log((1+SINR[i]), 2))
        print("The SINR is: "+str(SINR))
        #print(s_[0])
        ues_thr = s_[0]
        #print(action_to_ues_tbl[action][0])
        #print(ues_thr[action_to_ues_tbl[action][0]])

        print("The Rate is: "+str(R))
        ues_thr[action_to_ues_tbl[action][0]] += R[0]
        ues_thr[action_to_ues_tbl[action][1]] += R[1]

        s_ = np.array([ues_thr, channel_chain.next_state()], dtype=object)


        #print(s_)
        #print(observation)
        # self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        # reward function
        reward = 0
        for i in range (0, len(ues_thr)):
            reward = reward + float(np.log2(ues_thr[i]))

        if timer_tti == 100:
            done = True
        else:
            done = False
        return s_, reward, done

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

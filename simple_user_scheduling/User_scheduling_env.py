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
from itertools import combinations
from sympy import binomial
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
n_UEs = 3
comb = combinations(np.arange(n_UEs), 2)
action_to_ues_tbl = pd.Series(comb, index=np.arange(n_UEs))


class UserScheduling(object):
    def __init__(self):
        super(UserScheduling, self).__init__()
        #self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = binomial(n_UEs, 2)
        #self.title('User_scheduling')
        self.observations = np.ones((n_UEs,), dtype=int)
        #self._build_maze()

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

    def reset(self):
        self.observations = np.ones((n_UEs,), dtype=int)
        #self.update()
        #time.sleep(0.5)
        #self.canvas.delete(self.rect)
        #origin = np.array([20, 20])
        #self.rect = self.canvas.create_rectangle(
         #   origin[0] - 15, origin[1] - 15,
          #  origin[0] + 15, origin[1] + 15,
           # fill='red')
        # return observation
        return self.observations

    def step(self, action, observation, timer_tti):
        #s = self.canvas.coords(self.rect)
        s_ = observation.copy()
        base_action = np.array([0, 0])
        print(action_to_ues_tbl[action])
        print(action_to_ues_tbl[action][0])
        print(action_to_ues_tbl[action][1])
        s_[action_to_ues_tbl[action][0]] += action+1
        s_[action_to_ues_tbl[action][1]] += action+1

        print(s_)
        print(observation)
        #self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent



        # reward function
        reward = sum(np.log10(s_))
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
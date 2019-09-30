"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

import User_scheduling_env


class QLearningTable:
    def __init__(self, actions, learning_rate=0.8, reward_decay=0.95, e_greedy=0.1, max_epsilon=1.0, min_epsilon=0.01,
                 epsilon_decay=0.001):
        #self.file = open("test_6.txt", "w")
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.minimum_epsilon = min_epsilon
        self.maximum_epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, timer_tti, episode):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        if timer_tti == 100:
            #self.epsilon = self.minimum_epsilon + (self.maximum_epsilon - self.minimum_epsilon) * np.exp(
                #-self.epsilon_decay * episode)
            print(self.epsilon)
            print(episode)

            #self.test2()
            #if episode == 5:
                #self.test()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def test(self):
        #state = str(np.ones((3,), dtype=int))
        #while state in self.q_table.index:
            state = self.q_table[0]
            row = self.q_table.loc[0]
            maxvalue = self.q_table.loc[0, :].max()
            actions_state = self.q_table.loc[0]
            res = (actions_state == maxvalue)
            find_action = res.index[res.idxmax()]
            state = state.strip('[]')
            state_numpy = np.fromstring(state, dtype=int, sep=' ')
            tbl = User_scheduling_env.action_to_ues_tbl
            thr = find_action + 1
            state_numpy[tbl[find_action][0]] += thr
            state_numpy[tbl[find_action][1]] += thr
            self.file.write(state + ' ' + '[' + str(tbl[find_action][0]) + ',' + str(tbl[find_action][1]) + ']' + ' ' +
                            str(thr) + '\n')
            #state = str(state_numpy)
        #self.file.close()

    def test2(self):
        file = open("test_4.txt", "w")
        state = np.ones((3,), dtype=int)
        tbl = User_scheduling_env.action_to_ues_tbl
        max_pfs = 0
        j = 0
        while j < 100:

            for i in range(len(tbl)):
                r = i + 1
                avg_thr_1 = state[tbl[i][0]]
                avg_thr_2 = state[tbl[i][1]]
                pfs_1 = r / avg_thr_1
                pfs_2 = r / avg_thr_2
                if pfs_1 + pfs_2 >= max_pfs:
                    max_pfs = pfs_1 + pfs_2
                    tmp = i
            state[tbl[tmp][0]] += tmp + 1
            state[tbl[tmp][1]] += tmp + 1
            thr = tmp + 1
            file.write(str(state) + ' ' + '[' + str(tbl[tmp][0]) + ',' + str(tbl[tmp][1]) + ']' + ' ' +
                       str(thr) + '\n')
            j += 1
            max_pfs = 0
        file.close()

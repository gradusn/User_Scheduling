"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

import User_scheduling_env
import copy

import statistics


import csv


import MarkovChain

max_testing_episodes = 200000
count = 0
count_2 = 0
'''
Trying to solve the learning problem
reward_decay = 0.1
learning_rate = 0.1
start from 1,1 rates

qtable_SU_example_5tti_quant_noResetChan_rev05

10/3 - 14:08
Changed update of a chosen UE with thr = 1
ues_thr_rl[action] = (1 - (1 / time_window)) * ues_thr_rl[action] + (1 / time_window) * thr_rl
qtable_SU_example_5tti_quant_noResetChan_rev06
10/3 - 14:48
Changed to 10 slots 
qtable_SU_example_10tti_quant_noResetChan_rev0

12/3 18:34
Changed probablities to 0.95 transition
20Mil Iter
qtable_SU_example_10tti_quant_noResetChan_rev1

14/3 19:30
Added quant of GG
qtable_SU_example_10tti_quant2_rev0

17/3 9:07
Changed markov to 0.9, 0.1
qtable_SU_example_10tti_quant2_rev1

18/3 13:47
Changed Markov to 0.85, 0.15
qtable_SU_example_10tti_quant2_rev2

20/3 10:30
Changed Markov to 0.8,0.2
qtable_SU_example_10tti_quant2_rev3

21/3 17:09
Changed to 0.5,0.5
qtable_SU_example_10tti_quant2_rev4

'''

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.2, max_epsilon=1.0, min_epsilon=0.01,
                 epsilon_decay=0.000000007):
        #self.file = open("test_6.txt", "w")
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = max_epsilon
        self.minimum_epsilon = min_epsilon
        self.maximum_epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        #self.q_table = pd.read_pickle("qtable_SU_example_20tti_with_quant_with_tti.pkl")
    '''    
    def testing_markov(self, state, corr_state):
        actions_array = []
        q_learning_table = pd.read_pickle("q_learning_table_Markov_2_tti_1000000.pkl")
        channels = User_scheduling_env.create_channel(state, corr_state)
        create_rates = np.ones((User_scheduling_env.n_UEs,), dtype=float)
        create_observation = np.array([create_rates, channels[0]], dtype=object)
        actions = q_learning_table.loc[str(create_observation), :]
        s_ = copy.deepcopy(create_observation)
        rl_thr, actions_array = self.create_step(actions, env, s_, actions_array)
        
    '''

    def testing_markov(self, start_state, channel_chain, corr_chain, env, corr_state):
        q_learning_table = pd.read_pickle("q_learning_table_Markov_0.9_random_after_episode_2_tti_6000000.pkl")
        diff = []
        for episode in range(0, max_testing_episodes):
            print(episode)
            actions_array = []
            create_rates = np.ones((User_scheduling_env.n_UEs,), dtype=float)
            channels = env.create_channel(start_state, corr_state)
            create_observation = np.array([create_rates, channels], dtype=object)
            actions = q_learning_table.loc[str(create_observation), :]
            s_ = copy.deepcopy(create_observation)
            rl_thr, actions_array = self.create_step(actions, env, s_, actions_array)
            for i in range(1, User_scheduling_env.max_time_slots):
                start_state = channel_chain.next_state(start_state)
                channels = env.create_channel(start_state, corr_chain.next_state(0))
                create_observation = np.array([rl_thr, channels], dtype=object)
                actions = q_learning_table.loc[str(create_observation), :]
                s_ = copy.deepcopy(create_observation)
                rl_thr, actions_array = self.create_step(actions, env, s_, actions_array)
                if (i == User_scheduling_env.max_time_slots-1):
                    log_thrs = self.get_log_thrs(rl_thr)
                    #if log_thrs[0] > log_thrs[2]:
                    diff.append(log_thrs[0]-log_thrs[2])
                    #with open("Log_Thr_Markov_3_tti_test_0.9_epsilon_decay_20000000_with_ri_ti.csv", "a") as thr:
                        #thr_csv = csv.writer(thr, dialect='excel')
                        #thr_csv.writerow(log_thrs)
                        #thr.close()
                    #with open("actions_Markov_3_tti_test_0.9_epsilon_decay_20000000_with_ri_ti.csv", "a") as action_thr:
                        #action_csv = csv.writer(action_thr, dialect='excel')
                        #action_csv.writerow(actions_array)
                        #action_thr.close()
                    env.init_for_test()
                    actions_array = []
                    start_state = channel_chain.next_state(0)
                    corr_state = corr_chain.next_state(0)
        with open("Log_Thr_Markov_2_tti_test_0.9_epsilon_decay_60000000_200000_runs.csv", "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(diff)
            thr.close()

        #avg_run = statistics.mean(diff)
        #return avg_run

    def testing(self, states, env):
        actions_array = []
        q_learning_table = pd.read_pickle("q_learning_table_Markov_2_tti_4000000.pkl")
        for i in range(0, len(states)):
            channels = states[i]
            create_rates = np.ones((User_scheduling_env.n_UEs,), dtype=float)
            create_observation = np.array([create_rates, channels[0]], dtype=object)
            actions = q_learning_table.loc[str(create_observation), :]
            s_ = copy.deepcopy(create_observation)
            rl_thr, actions_array = self.create_step(actions, env, s_, actions_array)

            for j in range(1, len(channels)):
                create_observation = np.array([rl_thr, channels[j]], dtype=object)
                actions = q_learning_table.loc[str(create_observation), :]
                s_ = copy.deepcopy(create_observation)
                rl_thr, actions_array = self.create_step(actions, env, s_, actions_array)
                if (j == len(channels)-1):
                    log_thrs = self.get_log_thrs(rl_thr, env)
                    with open("Log_Thr_Markov_2_tti_4000000.csv", "a") as thr:
                        thr_csv = csv.writer(thr, dialect='excel')
                        thr_csv.writerow(log_thrs)
                        thr.close()
                    with open("actions_Markov_2_tti_4000000.csv", "a") as action_thr:
                        action_csv = csv.writer(action_thr, dialect='excel')
                        action_csv.writerow(actions_array)
                        action_thr.close()
                    env.init_for_test()
                    actions_array = []

    def get_log_thrs(self, rl_thr):
        sum_log_thrs = []
        thrs_algo = np.array([rl_thr, User_scheduling_env.ues_thr_random_global, User_scheduling_env.ues_thr_optimal_global, User_scheduling_env.ues_thr_ri_ti_global])
        for i in range(0, len(thrs_algo)):
            sum_log = 0
            thrs = thrs_algo[i]
            for j in range(0, len(thrs)):
                sum_log = sum_log + float(np.log2(thrs[j]))
            sum_log_thrs.append(sum_log)
        return sum_log_thrs

    def create_step(self, actions, env, s_, actions_array):
        choose_action = np.random.choice(actions[actions == np.max(actions)].index)
        optimal_action, R, tmp_thr_optimal, tmp_thr_ri_ti, action_ri_ti = env.get_rates(s_, choose_action, 'test')
        action_random = np.random.choice(self.actions)
        actions_array.append([choose_action, action_random, optimal_action, action_ri_ti])
        ues_thr_rl = s_[0]
        rl_thr = self.update_rates(R, ues_thr_rl, choose_action, optimal_action, tmp_thr_optimal, action_random, tmp_thr_ri_ti, action_ri_ti)

        return rl_thr, actions_array

    def update_rates(self, rates, ues_thr_rl, choose_action, optimal_action, tmp_thr_optimal, action_random, tmp_thr_ri_ti, action_ri_ti):
        for algo in User_scheduling_env.algorithm:
            if algo == 'rl':
                thr_rl = rates[choose_action]
                ues_thr_rl[User_scheduling_env.action_to_ues_tbl[choose_action][0]] += thr_rl[0]
                ues_thr_rl[User_scheduling_env.action_to_ues_tbl[choose_action][1]] += thr_rl[1]
            elif algo == 'random':
                thr_random = rates[action_random]
                User_scheduling_env.ues_thr_random_global[User_scheduling_env.action_to_ues_tbl[action_random][0]] += thr_random[0]
                User_scheduling_env.ues_thr_random_global[User_scheduling_env.action_to_ues_tbl[action_random][1]] += thr_random[1]
            elif algo == 'optimal':
                User_scheduling_env.ues_thr_optimal_global = tmp_thr_optimal
            else:
                User_scheduling_env.ues_thr_ri_ti_global = tmp_thr_ri_ti
        return ues_thr_rl


    def choose_action(self, observation, timer_tti):
        self.check_state_exist(observation, timer_tti)
        # action selection
        if np.random.uniform() > self.epsilon:

            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            if np.max(state_action) != 0:
                User_scheduling_env.best_action = 1
            else:
                User_scheduling_env.best_action = 0
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
            User_scheduling_env.best_action = 0

        return action

    def choose_action_test(self, observation):
        if observation not in self.q_table.index:
            print("No state in table")
            state_action = self.q_table.iloc[0, :]
        else:
            state_action = self.q_table.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)

        return action


    def learn(self, s, a, r, s_, timer_tti, episode, max_episodes):
        global count
        global count_2
        '''tTsS1988
        if timer_tti+1 == 3 and s_ not in self.q_table.index:
            count = count +1
        if timer_tti+1 == 2 and s_ not in self.q_table.index:
            count_2 = count_2+1
        '''
        self.check_state_exist(s_, timer_tti)
        q_predict = self.q_table.loc[str(s), a]
        check = s[1]
        '''
        if (check == 'G G 1'):
            enter = 1
        '''

        if (timer_tti < User_scheduling_env.max_time_slots):
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        # q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        '''
        if (q_target - q_predict) < 0 :
            enter = 2
        '''

        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)  # update
        self.epsilon = self.minimum_epsilon + (self.maximum_epsilon - self.minimum_epsilon) * np.exp(
            -self.epsilon_decay * episode)
        print(self.epsilon)

    def save_table(self, table1, table2, table3, table4, idxs):
        self.q_table.to_pickle("test_implementation_1.pkl")
        np.save("table_ue1_test_1.npy", table1)
        np.save("table_ue2_test_1.npy", table2)
        np.save("table_ue1_test_thr_1.npy", table3)
        np.save("table_ue2_test_thr_1.npy", table4)
        np.save("idx_1.npy", idxs)



    def load_table(self):
        self.q_table = pd.read_pickle("test_implementation_1.pkl")
        tmp_table_ue1 = np.load("table_ue1_test_1.npy")
        tmp_table_ue2 = np.load("table_ue2_test_1.npy")
        tmp_table_thr_ue1 = np.load("table_ue1_test_thr_1.npy", allow_pickle=True)
        tmp_table_thr_ue2 = np.load("table_ue2_test_thr_1.npy", allow_pickle=True)
        idx = np.load("idx_1.npy", allow_pickle=True)
        i = 0
        j = 0


        table_RL = np.full((100, 9), -1)
        table_2d_UE1 = np.full((len(tmp_table_ue1), len(tmp_table_thr_ue1)), -1)
        table_2d_UE2 = np.full((len(tmp_table_ue2), len(tmp_table_thr_ue2)), -1)


        for x in range(0, len(idx)):
            print(x)

            channels_ue1 = str(tmp_table_ue1[idx[x][0]])
            thr_ue1 = tmp_table_thr_ue1[idx[x][2]]

            channels_ue2 = str(tmp_table_ue2[idx[x][1]])
            thr_ue2 = tmp_table_thr_ue2[idx[x][3]]

            channels = channels_ue1 + " " + channels_ue2
            #tmp = tmp_table_thr_ue1[idx[i][0]][idx[i][3]]
            thr = np.array([thr_ue1, thr_ue2], dtype=int)
            find = str(np.array([thr, channels], dtype=object))

            if find in self.q_table.index:
                state_action = self.q_table.loc[find, :]
                #table_RL[idx[i][0]][idx[i][1]][idx[i][2]][idx[i][3]] = int(np.random.choice(state_action[state_action == np.max(state_action)].index))
                if (table_2d_UE1[idx[x][0]][idx[x][2]] != -1):
                    tmp_i = table_2d_UE1[idx[x][0]][idx[x][2]];
                    for j in range(0, 9):
                        if ( table_RL[tmp_i][j] != -1 ):
                            continue;
                        table_RL[tmp_i][j] = int(np.random.choice(state_action[state_action == np.max(state_action)].index))
                        table_2d_UE2[idx[x][1]][idx[x][3]] = j
                        break
                else:
                    if (table_2d_UE2[idx[x][1]][idx[x][3]] != -1):
                        tmp_j = table_2d_UE2[idx[x][1]][idx[x][3]]
                        for i in range(0, 100):
                            if (table_RL[i][tmp_j] != -1):
                                continue;
                            table_RL[i][tmp_j] = int(np.random.choice(state_action[state_action == np.max(state_action)].index))
                            table_2d_UE1[idx[x][0]][idx[x][2]] = i
                            break
                        #i = i + 1
                    else:
                        for i in range(0, 100):
                            for j in range(0,9):
                                if (table_RL[i][j] == -1):
                                    find = True
                                    break
                            if find:
                                break
                        table_RL[i][j] = int(np.random.choice(state_action[state_action == np.max(state_action)].index))
                        table_2d_UE1[idx[x][0]][idx[x][2]] = i
                        table_2d_UE2[idx[x][1]][idx[x][3]] = j






        np.savetxt("channel_ue1_1.csv", tmp_table_ue1, delimiter=',' , fmt='%s')
        np.savetxt("channel_ue2_1.csv", tmp_table_ue2, delimiter=',' , fmt='%s')
        np.savetxt("thr_ue1_1.csv", tmp_table_thr_ue1, delimiter=',' , fmt='%s')
        np.savetxt("thr_ue2_1.csv", tmp_table_thr_ue2, delimiter=',', fmt='%s')





        np.savetxt("table_RL_1.csv", table_RL, delimiter=',')

        np.savetxt("table_2d_UE1_1.csv", table_2d_UE1, delimiter=',')
        np.savetxt("table_2d_UE2_1.csv", table_2d_UE2, delimiter=',')

        #print(table_RL)



    def check_state_exist(self, state, timer_tti):
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

"""
Reinforcement learning simple user scheudling.

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
from matplotlib.ticker import PercentFormatter

from User_scheduling_env import UserScheduling
from Brain_user_scheduling import QLearningTable
from MarkovChain import MarkovChain
import User_scheduling_env


import timeit
import copy
import itertools
import numpy as np
import csv
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

option = 'train'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9

n_UEs = 2


property_to_probability1 = {'G': [1, 0], 'B': [0, 1]}
property_to_probability2 = {'G': [0.5, 0.5], 'B': [0.5, 0.5]}
property_to_probability3 = {'G': [0.1, 0.9], 'B': [0.1, 0.9]}


corr_probability = 0.8

max_episodes = 25000000
max_runs_stats = 500
max_test = 100000
Thr_convergence = []



def update():
    global state_action
    global start_state
    timer_tti = 1
    start_state = 'G G'
    channels = env.create_channel(start_state, timer_tti)
    observation = env.reset(channels, 0)
    User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)

    for episode in range(1, max_episodes):

        print("train " + str(episode))


        # RL choose action based on observation
        action = RL.choose_action(str(observation), timer_tti)


        # RL take action and get next observation and reward
        observation_, reward, start_state, done, finish_5000 = env.step(action, observation, start_state, timer_tti, channel_chain, episode)

        # RL learn from this transition
        RL.learn(observation, action, reward, str(observation_), timer_tti, episode, max_episodes)

        # swap observation

        observation = observation_

        timer_tti += 1


        if done:
            timer_tti = 1
            User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
            if episode % 3000 == 0:
                testing_convergence()
        if finish_5000 == 1:
            break

    # end of game
    User_scheduling_env.q_table_rl.to_pickle("qtable_SU_example_5tti_convergence_rl_UE1GUE20703.pkl")
    User_scheduling_env.q_table_pf.to_pickle("qtable_SU_example_5tti_convergence_pf_UE1GUE20703.pkl")
    User_scheduling_env.q_table_rr.to_pickle("qtable_SU_example_5tti_convergence_rr_UE1GUE20703.pkl")

    RL.save_table()
    print('training over')

def testing_convergence():
    global Thr_convergence
    for iter in User_scheduling_env.channels_array:
        channels = iter
        #channels = channels[0]
        channels = channels.split("_")
        User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
        User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)
        timer_tti = 1
        start_state = channels[0]
        channels_test = env.create_channel(start_state, timer_tti)
        observation = env.reset(channels_test, 1)
        for i in range(1, User_scheduling_env.max_time_slots + 1):
            if timer_tti == User_scheduling_env.max_time_slots:
                next_state = ""
            else:
                next_state = channels[i]
            action = RL.choose_action_test(str(observation))
            observation_, done = env.step_test_convergence(action, observation, timer_tti, next_state)
            observation = observation_
            timer_tti += 1
            if done:
                timer_tti = 1
                User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
                User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)
                env.add_channels_to_array(iter, 1)

    User_scheduling_env.channels_array = []

def plots():
    table_rl = pd.read_pickle("qtable_SU_example_5tti_convergence_rl_UE1GUE20505.pkl")
    table_rl_07 = pd.read_pickle("qtable_SU_example_5tti_convergence_rl_UE1GUE20703.pkl")
    table_pf = pd.read_pickle("qtable_SU_example_5tti_convergence_pf_UE1GUE20703.pkl")
    table_rr = pd.read_pickle("qtable_SU_example_5tti_convergence_rr_UE1GUE20703.pkl")


    convegrence_rl_07 = []
    convegrence_rl_05 = []
    convegrence_pf = []

    table_rl.drop(table_rl.columns[[0]], axis=1, inplace=True)
    for j in range(0, len(table_rl)):
        table_rl.iloc[j, :] = table_rl.iloc[j, :].replace(0, np.nan).ffill()
        table_rl_07.iloc[j, :] = table_rl_07.iloc[j, :].replace(0, np.nan).ffill()
        #print(table_rl.loc[table_rl.index[j], :][10000])
        convegrence_rl_05.append(table_rl.loc[table_rl.index[j], :][10000])
        convegrence_pf.append(table_pf.loc[table_rl.index[j], :][0])

    for j in range(0, len(table_rl)):
        convegrence_rl_07.append(table_rl_07.loc[table_rl.index[j], :][10000])
    '''
    for i in range(0, 10000):
        Avg.append(table_rl.iloc[:, i].mean())
    '''
    np.savetxt("convegrence_pf.csv", convegrence_pf, delimiter=",")
    #np.savetxt("convegrence_rl_05.csv", convegrence_rl_05, delimiter=",")
    #np.savetxt("convegrence_rl_07.csv", convegrence_rl_07, delimiter=",")


def test():
    global state_action
    global start_state

    start_state = 'G G'
    timer_tti = 1

    channels = env.create_channel(start_state, timer_tti)
    observation = env.reset(channels)
    User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 1, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global = np.full((1, n_UEs), 0.00001, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 1, dtype=float)
    RL.load_table()
    for iter in range(0, 1):
        string_pf = "q_learning_SU_10tti_pf_50RB_diff_gains_win10" + str(
            iter) + ".csv"
        string_rl = "q_learning_SU_10tti_rl_gb_quant2_0_5.csv"
        string_pf_short = "q_learning_SU_10tti_pf_gb_quant2_0_5.csv"
        string_gg = "GG_comaprison_0_5.csv"

        for episode in range(max_test):
            print("test " + str(episode))

            action = RL.choose_action_test(str(observation))
            observation_, start_state, done, finish_test = env.step_test(action, observation, start_state, timer_tti, channel_chain, episode)

            # swap observation
            observation = observation_
            timer_tti += 1

            if done:
                timer_tti = 1
                User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 1, dtype=float)
                User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 1, dtype=float)
            if finish_test == 1:
                break;

        print('testing ' + str(iter) + ' over')

        count_GG_rl = User_scheduling_env.count_GG_rl
        count_GG_pf = User_scheduling_env.count_GG_pf
        array_GG = [count_GG_rl, count_GG_pf]
        observations = User_scheduling_env.q_table

        with open(string_rl, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_rl)
            thr.close()
        with open(string_pf_short, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_pf_short)
            thr.close()
        with open(string_gg, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(array_GG)
            thr.close()



def test_markov():
    start_state = np.random.choice(states)
    corr_state = np.random.choice(corr)
    env.init_for_test()
    #RL.testing_markov(start_state, channel_chain, corr_chain, env, corr_state)
    data = np.genfromtxt('Log_Thr_2_tti_test_0.9_epsilon_decay_6000000_NN_SM.csv', delimiter = ',')
    sns.set(color_codes=True)
    sns.distplot(data, kde=False)
    #plt.hist(data, weights=np.ones(len(data)) / len(data))

    #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()



def Create_transtion_matrix(states):
    global property_to_probability1
    global property_to_probability2
    global property_to_probability3


    global_transition = [property_to_probability1, property_to_probability2]
    transition_matrix = []
    row_transition_matrix = []
    probability = 1
    for i in states:
        row_transition_matrix = []
        for j in states:
            probability = 1
            channels_1 = i.split()
            channels_2 = j.split()
            for k in range(0, len(channels_1)):
                if channels_1[k] == channels_2[k]:
                    probability = probability * global_transition[k][channels_1[k]][0]
                else:
                    probability = probability * global_transition[k][channels_1[k]][1]
            row_transition_matrix.append(probability)
        transition_matrix.append(row_transition_matrix)

    return transition_matrix











if __name__ == "__main__":




    states = ['G G G',
              'G G B',
              'G B G',
              'B G G',
              'G B B',
              'B B G',
              'B G B',
              'B B B'
              ]
    states_2_ues = ['G G',
              'G B',
              'B G',
              'B B',
              ]



    #corr = ['GB', 'BG', 'BB']

    #transition_matrix_corr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #corr_chain = MarkovChain(transition_matrix=transition_matrix_corr, states=corr)
    transition_matrix_channel = Create_transtion_matrix(states_2_ues)
    channel_chain = MarkovChain(transition_matrix=transition_matrix_channel,
                                states=states_2_ues)


    env = UserScheduling()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    #update()
    plots()
    #test()
    #test_markov()
    #env.after(100, update)
    #env.mainloop()
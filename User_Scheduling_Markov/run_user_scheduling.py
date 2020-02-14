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

import seaborn as sns
import matplotlib.pyplot as plt

option = 'train'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9

n_UEs = 2


property_to_probability1 = {'G': [1, 0], 'B': [0, 1]}
property_to_probability2 = {'G': [0, 1], 'B': [1, 0]}
property_to_probability3 = {'G': [0, 1], 'B': [1, 0]}


corr_probability = 0.8

max_episodes = 5000000
max_runs_stats = 500
max_test = 100000



def update():
    global state_action
    global start_state
    timer_tti = 1
    #start_state = 'G G G'
    start_state = states_simple[timer_tti - 1]
    channels = env.create_channel(start_state)
    observation = env.reset(channels)
    for episode in range(max_episodes):
        print("train " + str(episode))


        # RL choose action based on observation
        action = RL.choose_action(str(observation), timer_tti)


        # RL take action and get next observation and reward
        observation_, reward, start_state, done = env.step(action, observation, start_state, timer_tti, channel_chain, episode, states_simple)

        # RL learn from this transition
        RL.learn(str(observation), action, reward, str(observation_), timer_tti, episode, max_episodes)

        # swap observation

        observation = observation_

        timer_tti += 1


        if done:
            timer_tti = 1


    # end of game
    RL.save_table()
    print('training over')

def test():
    global state_action
    global start_state
    timer_tti = 1

    #start_state = 'G G G'
    start_state = states_simple[timer_tti - 1]
    channels = env.create_channel(start_state)
    observation = env.reset(channels)
    User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0.00001, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global = np.full((1, n_UEs), 0.00001, dtype=float)
    RL.load_table()
    for iter in range(0, 1):
        string_pf = "q_learning_SU_simple_10tti_pf_" + str(
            iter) + ".csv"
        string_rl = "q_learning_SU_simple_10ti_rl_50RB_win10_" + str(
            iter) + ".csv"
        string_pf_short = "q_learning_SU_simple_tti_pf_50RB_win10_Every2000_" + str(
            iter) + ".csv"

        for episode in range(max_test):
            print("test " + str(episode))

            action = RL.choose_action_test(str(observation), timer_tti)
            observation_, start_state, done = env.step_test(action, observation, start_state, timer_tti, channel_chain, episode, states_simple)

            # swap observation
            observation = observation_

            timer_tti += 1

            if done:
                #timer_tti = 1
                break

        print('testing ' + str(iter) + ' over')

        with open(string_rl, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_rl)
            thr.close()
        with open(string_pf_short, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_pf_short)
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


    global_transition = [property_to_probability1, property_to_probability2, property_to_probability3]
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

    states_simple = ['G2 G2',
              'B G0',
              'G0 G0',
              'G0 G3',
              'B B',
              'G0 B',
              'G1 G4',
              'G2 G1',
              'G0 B',
              'G2 G3'
              ]



    #corr = ['GB', 'BG', 'BB']

    #transition_matrix_corr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #corr_chain = MarkovChain(transition_matrix=transition_matrix_corr, states=corr)
    transition_matrix_channel = Create_transtion_matrix(states)
    channel_chain = MarkovChain(transition_matrix=transition_matrix_channel,
                                states=states)


    env = UserScheduling()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update()
    #test()
    #test_markov()
    #env.after(100, update)
    #env.mainloop()
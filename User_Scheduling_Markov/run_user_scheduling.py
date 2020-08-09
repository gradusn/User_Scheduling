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

n_UEs = 3


property_to_probability1 = {'G': [0.8, 0.2], 'B': [0.2, 0.8]}
property_to_probability2 = {'G': [0.2, 0.8], 'B': [0.8, 0.2]}
property_to_probability3 = {'G': [0.5, 0.5], 'B': [0.5, 0.5]}


corr_probability = 0.8

max_episodes = 50000000
max_runs_stats = 500
max_test = 100000
table_UE1 = []
table_UE2 = []
table_UE3 = []




def update():
    global state_action
    global start_state
    timer_tti = 1
    start_state = 'G G'
    channels = env.create_channel(start_state, timer_tti)
    observation = env.reset(channels)
    '''
    if int(observation[0][0]) not in table_UE1:
        table_UE1.append(int(observation[0][0]))
    if int(observation[0][1]) not in table_UE2:
        table_UE2.append(int(observation[0][1]))
    if int(observation[0][2]) not in table_UE3:
        table_UE3.append(int(observation[0][2]))
    '''
    for episode in range(max_episodes):
        print("train " + str(episode))

        # RL choose action based on observation
        #states_to_table = str(int(observation[0][0]))+ " "+str(int(observation[0][1]))+" "+str(int(observation[0][2])) + " " + str(observation[1])
        #states_to_table = str(int(observation[0][0]))+ " "+str(int(observation[0][1])) + " " + str(observation[1])

        action = RL.choose_action(str(observation), timer_tti)

        # RL take action and get next observation and reward
        observation_, reward, start_state, done = env.step(action, observation, start_state, timer_tti, channel_chain, episode)

        # RL learn from this transition
        '''
        if int(observation_[0][0]) not in table_UE1:
            table_UE1.append(int(observation_[0][0]))
        if int(observation_[0][1]) not in table_UE2:
            table_UE2.append(int(observation_[0][1]))
        if int(observation_[0][2]) not in table_UE3:
            table_UE3.append(int(observation_[0][2]))
        '''
        #states_to_table_2 = str(int(observation_[0][0])) + " "+str(int(observation_[0][1])) + " " + str(observation_[1])

        #states_to_table_2 = str(int(observation_[0][0])) + " "+str(int(observation_[0][1]))+" "+str(int(observation[0][2]))  + " " + str(observation_[1])

        RL.learn(observation, action, reward, str(observation_), timer_tti, episode, max_episodes)

        # swap observation

        observation = observation_

        timer_tti += 1


        if done:
            timer_tti = 1

    # end of game
    RL.save_table(table_UE1, table_UE2)
    print('training over')

def test():
    global state_action
    global start_state

    start_state = 'G G'
    timer_tti = 1

    channels = env.create_channel(start_state, timer_tti)
    observation = env.reset(channels)
    User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global_noavg = np.full((1, n_UEs), 0, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)
    RL.load_table()
    for iter in range(0, 1):
        string_pf = "q_learning_SU_5tti_pf_1_noquant_UE1G0802_UE2G0208.csv"
        string_rl = "q_learning_SU_5tti_rl_1_noquant_UE1G0802_UE2G0208.csv"
        string_rr = "qq_learning_SU_5tti_rr_1_noquant_UE1G0802_UE2G0208.csv"

        for episode in range(max_test):
            print("test " + str(episode))
            #states_to_table = str(int(observation[0][0])) + " " + str(int(observation[0][1])) + " " + str(
                #observation[1])

            #action = RL.choose_action_test(str(observation))
            action = RL.choose_action_test(str(observation))
            observation_, start_state, done = env.step_test(action, observation, start_state, timer_tti, channel_chain, episode)

            # swap observation
            observation = observation_
            timer_tti += 1

            if done:
                timer_tti = 1
                User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
                User_scheduling_env.ues_thr_ri_ti_global_noavg = np.full((1, n_UEs), 0, dtype=float)
                User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)

        print('testing ' + str(iter) + ' over')

        with open(string_rl, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_rl)
            thr.close()
        with open(string_pf, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_pf_short)
            thr.close()
        with open(string_rr, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.metric_rr)
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


    #global_transition = [property_to_probability1, property_to_probability2, property_to_probability3]
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
    test()
    #test_markov()
    #env.after(100, update)
    #env.mainloop()
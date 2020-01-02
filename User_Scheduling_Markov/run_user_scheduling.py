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
property_to_probablity = {'G': [alpha_GB, 1-alpha_GB], 'B': [beta_GB, 1 - beta_GB]}
corr_probability = 0.8

max_episodes = 10000000
max_runs_stats = 500
max_test = 500000



def update():
    global state_action
    global start_state
    for episode in range(max_episodes):
        print("train " + str(episode))

        timer_tti = 0
        # initial observation
        #start_state = channel_chain.next_state(start_state)
        start_state = np.random.choice(states)
        channels = env.create_channel(start_state, corr_chain.next_state(0))
        observation, state = env.reset(channels)



        while True:
            timer_tti += 1
            # fresh env
            #env.render()


            # RL choose action based on observation
            action = RL.choose_action(str(observation), timer_tti)


            # RL take action and get next observation and reward
            observation_, reward, start_state, done, state_ = env.step(action, observation, corr_chain, start_state, timer_tti, channel_chain, episode, state)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), timer_tti, episode, max_episodes)

            # swap observation

            observation = observation_
            state = state_

            if done:
                break

    # end of game
    RL.save_table()
    print('training over')

def test():
    global state_action
    global start_state
    RL.load_table()
    for iter in range(0, 1):
        string = "Log_Thr_1000_tti_epsilon_decay_100000000_NN_SU_3_ues_no_max_tti_riti_same_rayleigh" + str(
            iter) + ".csv"

        for episode in range(max_test):
            print("test " + str(episode))

            timer_tti = 0
            start_state = np.random.choice(states)
            channels = env.create_channel(start_state, corr_chain.next_state(0))
            observation = env.reset(channels)

            while True:
                timer_tti += 1

                action = RL.choose_action_test(str(observation))
                observation_, start_state, done = env.step_test(action, observation, corr_chain, start_state, timer_tti, channel_chain, episode)

                # swap observation
                observation = observation_

                if done:
                    break

        print('testing ' + str(iter) + ' over')
        with open(string, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(User_scheduling_env.diff)
            thr.close()
        User_scheduling_env.diff = []



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
    global property_to_probablity
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
                    probability = probability * property_to_probablity[channels_1[k]][0]
                else:
                    probability = probability * property_to_probablity[channels_1[k]][1]
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

    corr = ['GB', 'BG', 'BB']

    transition_matrix_corr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    corr_chain = MarkovChain(transition_matrix=transition_matrix_corr, states=corr)
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
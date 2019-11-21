from Enviroment_DQN import UserScheduling
import Enviroment_DQN
from DQN_brain import DeepQNetwork
from MarkovChain import MarkovChain


import copy
import itertools
import numpy as np
import csv

import seaborn as sns
import matplotlib.pyplot as plt

max_episodes = 6000000
state_action = []

alpha_GB = 0.9
beta_GB = 0.9
property_to_probablity = {'G': [alpha_GB, 1-alpha_GB], 'B': [beta_GB, 1 - beta_GB]}
option = 'train'
start_test = 200000


def run_DQN():
    step = 0
    global option
    for episode in range(max_episodes):
        print(episode)
        # scheduling period
        timer_tti = 0
        # initial observation
        start_state = np.random.choice(states)
        channels = env.create_channel(start_state, corr_chain.next_state(0))
        if episode > max_episodes - start_test:
            option = 'test'
        observation = env.reset(channels)

        while True:
            timer_tti += 1

            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward

            observation_, reward, done, start_state = env.step(action, observation, corr_chain, start_state, timer_tti, channel_chain, option)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn(timer_tti, episode)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            step += 1
            if done:
            #print(self.epsilon)
                break


    # end of game
    print('game over')
    #env.destroy()

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
    #States for channel gains
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
    # maze game
    env = UserScheduling()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #env.after(100, run_maze)
    #env.mainloop()
    run_DQN()
    with open("Log_Thr_2_tti_test_0.9_epsilon_decay_2000000_NN_SM.csv", "a") as thr:
        thr_csv = csv.writer(thr, dialect='excel')
        thr_csv.writerow(Enviroment_DQN.diff)
        thr.close()

    RL.plot_cost()
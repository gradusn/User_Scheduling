"""
Reinforcement learning simple user scheudling.

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from User_scheduling_env import UserScheduling
from Brain_user_scheduling import QLearningTable
from MarkovChain import MarkovChain

import timeit
import copy
import itertools


option = 'train'
state_action = []

alpha_GB = 0.2
beta_GB = 1-alpha_GB
property_to_probablity = {'G': alpha_GB, 'B': beta_GB}


def update():
    global state_action
    for episode in range(200000):
        timer_tti = 0
        # initial observation
        observation = env.reset(channel_chain.next_state())
        #observation = env.reset('B G B_BB')

        observation_old = copy.deepcopy(observation)

        while True:
            timer_tti += 1
            # fresh env
            #env.render()

            state_action_old = copy.deepcopy(state_action)

            # RL choose action based on observation
            action, state_action = RL.choose_action(str(observation), timer_tti)


            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action, observation, timer_tti, channel_chain, episode, observation_old, option, state_action, state_action_old)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), timer_tti, episode, observation_old)

            # swap observation
            observation_old = copy.deepcopy(observation)

            observation = observation_


            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('training over')
    #env.destroy()

def test():
    global option
    option = 'test'
    states = ['G G G_GB', 'G G G_BG', 'G G G_BB',
              'G G B_GB', 'G G B_BG', 'G G B_BB',
              'G B G_GB', 'G B G_BG', 'G B G_BB',
              'B G G_GB', 'B G G_BG', 'B G G_BB',
              'G B B_GB', 'G B B_BG', 'G B B_BB',
              'B B G_GB', 'B B G_BG', 'B B G_BB',
              'B G B_GB', 'B G B_BG', 'B G B_BB',
              'B B B_GB', 'B B B_BG', 'B B B_BB'
              ]
    states_test = [('B G B_BG', 'B B G_BG')]

    states_possible = [p for p in itertools.product(states, repeat=2)]
    env.init_for_test()
    RL.testing(states_possible, env)
    #example = [x for x in itertools.product([1,2,3], repeat=2)]
    #size = len(states_possible)
    #print(size)

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
                    probability = probability * property_to_probablity[channels_1[k]]
                else:
                    probability = probability * property_to_probablity[channels_2[k]]
            row_transition_matrix.append(probability)
        transition_matrix.append(row_transition_matrix)

    return transition_matrix











if __name__ == "__main__":


    alpha_GB = 0.5
    Beta_GB = 1-alpha_GB

    states = ['G G G',
              'G G B',
              'G B G',
              'B G G',
              'G B B',
              'B B G',
              'B G B',
              'B B B'
              ]

    transition_matrix_channel = Create_transtion_matrix(states)



    channel_chain = MarkovChain(transition_matrix=transition_matrix_channel,
                                states=states)





    env = UserScheduling()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update()
    #test()
    #env.after(100, update)
    #env.mainloop()
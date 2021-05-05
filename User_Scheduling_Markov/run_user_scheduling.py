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
import time

import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

option = 'train'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9

n_UEs = 4
#n_UEs = 2

property_to_probability1 = {'G': [1, 0], 'B': [0, 1]}
property_to_probability2 = {'G': [0.3, 0.7], 'B': [0.7, 0.3]}
property_to_probability3 = {'G': [0, 1], 'B': [1, 0]}
property_to_probability4 = {'G': [1, 0], 'B': [0, 1]}
property_to_probability5 = {'G': [1, 0], 'B': [0, 1]}




corr_probability = 0.8
Reset_num = 1000
max_episodes = 50000000
max_runs_stats = 500
max_test = 10000
table_UE1 = []
table_UE2 = []
table_UE3 = []





def update():
    global state_action
    global start_state
    timer_tti = 1
    start_state = 'G G B G'
    #start_state = 'G G G B G'
    #start_state = 'G G'
    '''
    with open('iTBS_UE0_1.csv', newline='') as csvfile:
        UE1_ITBS = csv.reader(csvfile)
        UE1_ITBS = list(UE1_ITBS)
    '''
    start_time = time.time()
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
    print("%s Seconds of train" % (time.time() - start_time))

    print('training over')

def test():
    global state_action
    global start_state

    start_state = 'G G'
    #start_state = 'G B G B'
    #start_state = 'G G B'
    timer_tti = 1
    timer_tti_for_reset = 1
    reset = Reset_num/100


    channels = env.create_channel(start_state, timer_tti)
    observation = env.reset(channels)
    User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global_noavg = np.full((1, n_UEs), 0, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)
    User_scheduling_env.ues_thr_ri_ti_global_short_accum_thr = np.full((1, n_UEs), 0, dtype=float)
    User_scheduling_env.jitter_slots = np.full((1, n_UEs), -1, dtype=float). flatten()
    User_scheduling_env.jitter_slots_pf = np.full((1, n_UEs), -1, dtype=float). flatten()

    RL.load_table()
    for iter in range(0, 1):
        string_pf = "q_learning_SU_6tti_pf_1_noquant_UE1GUE2B0703.csv"
        string_rl = "q_learning_SU_6tti_rl_1_noquant_UE1GUE2B0703.csv"
        string_rr = "q_learning_5_tti_rr_3Ues.csv"

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
            timer_tti_for_reset += 1

            if done:
                timer_tti = 1
                #User_scheduling_env.ues_thr_ri_ti_global_noavg = np.full((1, n_UEs), 0, dtype=float)
                #User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)
            if timer_tti_for_reset > reset:
                User_scheduling_env.ues_thr_ri_ti_global_rr = np.full((1, n_UEs), 0, dtype=float)
                User_scheduling_env.ues_thr_ri_ti_global_short = np.full((1, n_UEs), 0, dtype=float)
                User_scheduling_env.ues_thr_ri_ti_global_short_accum_thr = np.full((1, n_UEs), 0, dtype=float)
                timer_tti_for_reset = 1
        print('testing ' + str(iter) + ' over')

        #f = open("results_3UEs_5TTi.txt", "a")
        #f = open("results_2UEs_5TTi_0505.txt", "a")
        '''
        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_gcount_rl.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.list_g)
        f.close()

        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_bcount_rl.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.list_b)
        f.close()

        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_bcount_pf.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.list_b_pf)
        f.close()

        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_gcount_pf.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.list_g_pf)
        f.close()
        
        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_rl_jitter_UE3.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.jitter_UE3)
        f.close()

        with open('RResults_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_pf_jitter_UE1.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.jitter_UE1_pf)
        f.close()

        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_pf_jitter_UE2.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.jitter_UE2_pf)
        f.close()

        with open('Results_qtable_SU_example_10tti_UE1G_UE2B0703_UE3B_pf_jitter_UE3.txt', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(User_scheduling_env.jitter_UE3_pf)
        f.close()
        '''

        with open('Results_qtable_10tti_UE1G_UE2B0703' + str(int(reset)) + 'reset_optimal.txt', 'w') as f:
            for list in User_scheduling_env.arr_accum_thr_optimal:
                f.write(str(list)[1:-1] +'\n')
        f.close()

        with open('Results_qtable_10tti_UE1G_UE2B0703' + str(int(reset)) + 'reset_rr.txt', 'w') as f:
            for list in User_scheduling_env.arr_accum_thr_rr:
                f.write(str(list)[1:-1] +'\n')
        f.close()

        with open('Results_qtable_10tti_UE1G_UE2B0703' + str(int(reset)) + 'reset_rl.txt', 'w') as f:
            for list in User_scheduling_env.arr_accum_thr_rl:
                f.write(str(list)[1:-1] +'\n')
        f.close()

        with open('Results_qtable_10tti_UE1G_UE2B0703' + str(int(reset)) + 'reset_pf.txt', 'w') as f:
            for list in User_scheduling_env.arr_accum_thr_pf:
                f.write(str(list)[1:-1] +'\n')
        f.close()



        avg_rl = np.array(User_scheduling_env.metric_rl_accum_thr).mean()
        avg_pf = np.array(User_scheduling_env.metric_pf_short_accum_thr).mean()

        to_write = str(avg_rl) + "  " + str(avg_pf)
        f.write(to_write)
        f.close()

        arr_rl = np.asarray(User_scheduling_env.metric_rl)[np.newaxis]
        arr_rl = np.transpose(arr_rl)

        arr_pf = np.asarray(User_scheduling_env.metric_pf_short)[np.newaxis]
        arr_pf = np.transpose(arr_pf)

        arr_rr = np.asarray(User_scheduling_env.metric_rr)[np.newaxis]
        arr_rr = np.transpose(arr_rr)

        with open(string_rr, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerows(arr_rr)
            thr.close()

        with open(string_rl, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerows(arr_rl)
            thr.close()
        with open(string_pf, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerows(arr_pf)
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
    global property_to_probability4
    global property_to_probability5


    #global_transition = [property_to_probability1, property_to_probability2, property_to_probability3,property_to_probability4, property_to_probability4]


    global_transition = [property_to_probability1, property_to_probability2, property_to_probability3, property_to_probability4]
    #global_transition = [property_to_probability1, property_to_probability2]

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
    states_4Ues = ['G G G G',
                   'G G G B',
                   'G G B G',
                   'G B G G',
                   'B G G G',
                   'G G B B',
                   'G B B G',
                   'B B G G',
                   'B G B G',
                   'G B G B',
                   'B G G B',
                   'B B B G',
                   'G B B B',
                   'B B G B',
                   'B G B B',
                   'B B B B']
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

    a = list(product(['G' ,'B'], repeat=4))
    test_states = []
    for i in a:
        test_states.append(' '.join(i))

    #corr = ['GB', 'BG', 'BB']

    #transition_matrix_corr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #corr_chain = MarkovChain(transition_matrix=transition_matrix_corr, states=corr)
    transition_matrix_channel = Create_transtion_matrix(test_states)
    channel_chain = MarkovChain(transition_matrix=transition_matrix_channel,
                                states=test_states)


    env = UserScheduling()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    update()
    #test()
    #test_markov()
    #env.after(100, update)
    #env.mainloop()
from enviroment_DQN import UserScheduling
from brain_DQN import DeepQNetwork
import enviroment_DQN
import numpy as np
import csv
import time

from MarkovChain import MarkovChain



import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist

option = 'train'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9
property_to_probability1 = {'G': [1, 0], 'B': [0, 1]}
property_to_probability2 = {'G': [0.1, 0.9], 'B': [0.9, 0.1]}
property_to_probability3 = {'G': [0.1, 0.9], 'B': [0.1, 0.9]}
n_UEs = 2

max_episodes = 10000000
max_test = 150000

def update():
    step = 0
    global state_action
    global start_state
    start_time = time.time()
    timer_tti = 0
    start_state = 'G G'
    start_state_snr = env.create_channel(start_state)
    observation = env.reset(start_state_snr)

    for episode in range(max_episodes):
        print("train " + str(episode))

        timer_tti += 1

        # RL choose action based on observation
        action = RL.choose_action(observation)
        # RL take action and get next observation and reward
        observation_, reward, done, next_channel_state = env.step(action, observation, start_state, episode, timer_tti, channel_chain)

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn(episode, max_episodes, timer_tti)

        # swap observation
        observation = observation_
        start_state = next_channel_state

        # break while loop when end of this episode
        step += 1

        if done:
            timer_tti = 0

    # end of game
    RL.save_mode()
    print('training over')
    print("--- %s seconds of training ---" % (time.time() - start_time))

def test():
    global state_action
    global start_state

    timer_tti = 0
    start_state = 'G G'
    start_state_snr = env.create_channel(start_state)
    observation = env.reset(start_state_snr)
    RL.load_model()
    enviroment_DQN.ues_thr_ri_ti_global = np.full((1, n_UEs), 0, dtype=float)
    enviroment_DQN.ues_thr_ri_ti_global_noavg = np.full((1, n_UEs), 0, dtype=float)

    for iter in range(0,1):
        string_pf = "DQN_SU_simple_5ti_pf_q09_p01_lr001_rd06_for_no_avg.csv"
        string_rl = "DQN_SU_simple_5tti_rl_q09_p01_lr001_rd06_for_no_avg.csv"

        for episode in range(max_test):

            timer_tti += 1
            print("test " + str(episode))
            # RL choose action based on observation
            action = RL.choose_action_test(observation)

            # RL take action and get next observation and reward
            observation_, done, next_channel_state = env.step_test(action, observation, start_state, episode, timer_tti, channel_chain)

            # swap observation
            observation = observation_
            start_state = next_channel_state

            if done:
                timer_tti = 0
                enviroment_DQN.ues_thr_ri_ti_global = np.full((1, n_UEs), 0, dtype=float)
                enviroment_DQN.ues_thr_ri_ti_global_noavg = np.full((1, n_UEs), 0, dtype=float)

        print('testing ' + str(iter) + ' over')
        with open(string_rl, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(enviroment_DQN.metric_rl)
            thr.close()
        with open(string_pf, "a") as thr:
            thr_csv = csv.writer(thr, dialect='excel')
            thr_csv.writerow(enviroment_DQN.metric_pf)
            thr.close()



def plot():


    iter = 0
    string = "Log_Thr_NN_model_3_ues_SU_DQN_10_tti_10_timeScale_1051rays" + str(iter) + ".csv"
    data = np.genfromtxt(string, delimiter=',')
    #data2d_1 = np.atleast_2d(data)
    for iter in range(1, 1):
        string = "Log_Thr_1000_tti_epsilon_decay_100000000_NN_SU_3_ues_no_max_tti_riti_same_rayleigh" + str(iter) + ".csv"
        data = np.genfromtxt(string, delimiter=',')
        data2d = np.atleast_2d(data)
        gather = np.concatenate((data2d_1, data2d), axis=0)
        data2d_1 = gather
   # mean_result = np.mean(data2d_1, axis=0)
    sns.set(color_codes=True)
    sns.distplot(data, kde=False)
    plt.xlabel("LogThr BF vs RL")
    plt.ylabel("Number of Observations")
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
    states_2_ues = ['G G',
                    'G B',
                    'B G',
                    'B B',
                    ]

    transition_matrix_channel = Create_transtion_matrix(states_2_ues)
    channel_chain = MarkovChain(transition_matrix=transition_matrix_channel,
                                states=states_2_ues)

    env = UserScheduling()

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.1,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    #update()
    test()
    #with open("Log_Thr_2_tti_test_0.9_epsilon_decay_60000000_NN_SU.csv", "a") as thr:
        #thr_csv = csv.writer(thr, dialect='excel')
        #thr_csv.writerow(enviroment_DQN.diff)
        #thr.close()


    #RL.plot_cost()

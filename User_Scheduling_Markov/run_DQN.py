from enviroment_DQN import UserScheduling
from brain_DQN import DeepQNetwork
import enviroment_DQN
import numpy as np
import csv



import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist

option = 'test'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9
property_to_probablity = {'G': [alpha_GB, 1-alpha_GB], 'B': [beta_GB, 1 - beta_GB]}
corr_probability = 0.8

max_episodes = 60000000
start_test = 200000

def update():
    step = 0
    global state_action
    global start_state

    start_state = env.create_rayleigh_fading()
    channes_guass_corr = env.create_guass_vectors()
    channels = env.create_channel(start_state, channes_guass_corr)
    observation = env.reset(channels)

    for episode in range(max_episodes):
        print("train " + str(episode))

        # RL choose action based on observation
        action = RL.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward = env.step(action, observation, start_state, episode)

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
                RL.learn(episode)

        # swap observation
        observation = observation_

         # break while loop when end of this episode
        step += 1

def test():
    global state_action
    global start_state

    start_state = env.create_rayleigh_fading()
    channes_guass_corr = env.create_guass_vectors()
    channels = env.create_channel(start_state, channes_guass_corr)
    observation = env.reset(channels)
    timer_tti = 0

    for episode in range(max_episodes):
        print("test " + str(episode))

        # RL choose action based on observation
        action = RL.choose_action_test(observation)

        # RL take action and get next observation and reward
        observation_ = env.step_test(action, observation, start_state, episode, timer_tti)

        # swap observation
        observation = observation_

        timer_tti += 1
        timer_tti = timer_tti % 10

    # end of game
    print('testing over')


if __name__ == "__main__":


    meanvalue = 3
    modevalue = np.sqrt(2 / np.pi) * meanvalue

    meanvalue1 = 2
    modevalue1 = np.sqrt(2 / np.pi) * meanvalue1

    meanvalue2 = 1
    modevalue2 = np.sqrt(2 / np.pi) * meanvalue2
    env = UserScheduling(modevalue, modevalue1, modevalue2)

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    #update()
    test()

    with open("Log_Thr_2_tti_test_0.9_epsilon_decay_60000000_NN_GM.csv", "a") as thr:
        thr_csv = csv.writer(thr, dialect='excel')
        thr_csv.writerow(enviroment_DQN.diff)
        thr.close()
    #RL.plot_cost()
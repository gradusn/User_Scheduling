from enviroment_DQN import UserScheduling
from brain_DQN import DeepQNetwork
import enviroment_DQN
import numpy as np
import csv



import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist

option = 'train'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9
property_to_probablity = {'G': [alpha_GB, 1-alpha_GB], 'B': [beta_GB, 1 - beta_GB]}
corr_probability = 0.8

max_episodes = 10
start_test = 5

def update():
    global option
    step = 0
    global state_action
    global start_state
    for episode in range(max_episodes):
        print(episode)

        timer_tti = 0
        # initial observation
        # start_state = channel_chain.next_state(start_state)
        # start_state = np.random.choice(states)
        start_state = env.create_rayleigh_fading()
        #channes_guass_corr = env.create_guass_vectors()
        #channels = env.create_channel(start_state, channes_guass_corr)
        #observation = env.reset(channels)
        observation = env.reset(start_state)
        if episode > max_episodes - start_test:
            option = 'test'

        while True:
            timer_tti += 1

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action, observation, start_state, timer_tti, episode, option )

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn(timer_tti, episode)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            step += 1

            if done:
                break

    # end of game
    print('game over')

def plot():
    data = np.genfromtxt('Log_Thr_2_tti_test_0.9_epsilon_decay_60000000_NN_GM.csv',
                         delimiter=',')
    sns.set(color_codes=True)
    sns.distplot(data, kde=False)
    plt.xlabel("LogThr BF vs RL")
    plt.ylabel("Number of Observations")
    plt.show()


if __name__ == "__main__":
    #plot()

    meanvalue = 3
    modevalue = np.sqrt(2 / np.pi) * meanvalue

    meanvalue1 = 2
    modevalue1 = np.sqrt(2 / np.pi) * meanvalue1

    meanvalue2 = 1
    modevalue2 = np.sqrt(2 / np.pi) * meanvalue2

    (n, bins0, patches) = hist(np.random.rayleigh(modevalue, 50000000), bins=16)
    (n1, bins1, patches1) = hist(np.random.rayleigh(modevalue1, 50000000), bins=16)
    (n2, bins2, patches2) = hist(np.random.rayleigh(modevalue2, 50000000), bins=16)

    #bins = [bins0, bins1, bins2]

    env = UserScheduling(modevalue, modevalue1, modevalue2, bins0)

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    update()
    #with open("Log_Thr_2_tti_test_0.9_epsilon_decay_60000000_NN_SU.csv", "a") as thr:
        #thr_csv = csv.writer(thr, dialect='excel')
        #thr_csv.writerow(enviroment_DQN.diff)
        #thr.close()


    #RL.plot_cost()

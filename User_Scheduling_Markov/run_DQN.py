from enviroment_DQN import UserScheduling
from brain_DQN import DeepQNetwork

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist

option = 'train'
state_action = []

alpha_GB = 0.9
beta_GB = 0.9
property_to_probablity = {'G': [alpha_GB, 1-alpha_GB], 'B': [beta_GB, 1 - beta_GB]}
corr_probability = 0.8

max_episodes = 20000000
max_runs_stats = 500

def update():
    step = 0
    global state_action
    global start_state
    for episode in range(max_episodes):
        timer_tti = 0
        # initial observation
        # start_state = channel_chain.next_state(start_state)
        # start_state = np.random.choice(states)
        start_state = env.create_rayleigh_fading()
        channes_guass_corr = env.create_guass_vectors()
        channels = env.create_channel(start_state, channes_guass_corr)
        observation = env.reset(channels)

        while True:
            timer_tti += 1

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action, observation, start_state, timer_tti, episode, option )

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            step += 1

            if done:
                break

    # end of game
    print('game over')


if __name__ == "__main__":
    env = UserScheduling()

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    RL.plot_cost()
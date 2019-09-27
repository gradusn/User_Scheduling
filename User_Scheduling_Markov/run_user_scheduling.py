"""
Reinforcement learning simple user scheudling.

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from User_scheduling_env import UserScheduling
from Brain_user_scheduling import QLearningTable
from MarkovChain import MarkovChain




def update():
    for episode in range(20000):
        timer_tti = 0
        # initial observation
        observation = env.reset(channel_chain.next_state())

        while True:
            timer_tti += 1
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action, observation, timer_tti, channel_chain)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), timer_tti, episode)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":


    transition_matrix = [[0.8, 0.19, 0.01],
                         [0.2, 0.7, 0.1],
                         [0.1, 0.2, 0.7]]

    channel_chain = MarkovChain(transition_matrix=transition_matrix,
                                states=['G G G_GB', 'G G G_BG', 'G G G_BB',
                                        'G G B_GB', 'G G B_BG', 'G G B_BB',
                                        'G B G_GB', 'G B G_BG', 'G B G_BB',
                                        'B G G_GB', 'B G G_BG', 'B G G_BB',
                                        'G B B_GB', 'G B B_BG', 'G B B_BB',
                                        'B B G_GB', 'B B G_BG', 'B B G_BB',
                                        'B G B_GB', 'B G B_BG', 'B G B_BB',
                                        'B B B_GB', 'B B B_BG', 'B B B_BB'
                                        ])



    env = UserScheduling()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update()
    #env.after(100, update)
    #env.mainloop()
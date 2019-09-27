"""
Reinforcement learning simple user scheudling.

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from Maze_new import Maze
from User_scheduling_env import UserScheduling
from Brain_user_scheduling import QLearningTable




def update():
    for episode in range(20000):
        timer_tti = 0
        # initial observation
        observation = env.reset()

        while True:
            timer_tti += 1
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action, observation, timer_tti)

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
    env = UserScheduling()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    update()
    #env.after(100, update)
    #env.mainloop()
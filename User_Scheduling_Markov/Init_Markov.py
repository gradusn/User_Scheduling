from MarkovChain import MarkovChain

if __name__ == "__main__":
    transition_matrix = [[0.8, 0.19, 0.01],
                         [0.2, 0.7, 0.1],
                         [0.1, 0.2, 0.7]]

    weather_chain = MarkovChain(transition_matrix=transition_matrix,
                                states=['GGG_GGG', 'GGG_BGG', 'GGG_GBG', 'GGG_GGB', 'GGG_GBB',
                                        'GGG_BBG', 'GGG_BGB', 'GGG_BBB',
                                        'GGB_GGG', 'GGB_BGG', 'GGB_GBG', 'GGB_GGB', 'GGB_GBB',
                                        'GGB_BGB', 'GGB_BBG', 'GGB_BBB',
                                        'GBG_GGG', 'GBG_BGG', 'GBG_GBG', 'GBG_GGB', 'GBG_GBB',
                                        'BGG_BBG', 'BGG_BGB', 'BGG_BBB',
                                        'BGG_GGG', 'BGG_BGG', 'BGG_GBG', 'BGG_GGB', 'BGG_GBB',
                                        'BGG_BBG', 'BGG_BGB', 'BGG_BBB',
                                        'GBB_GGG', 'GBB_BGG', 'GBB_GBG', 'GBB_GGB', 'GBB_GBB',
                                        'GBB_BBG', 'GBB_BGB', 'GBB_BBB',
                                        'BBG_GGG', 'BBG_BGG', 'BBG_GBG', 'BBG_GGB', 'BBG_GBB',
                                        'BBG_BBG', 'BBG_BGB', 'BBG_BBB',
                                        'BGB_GGG', 'BGB_BGG', 'BGB_GBG', 'BGB_GGB', 'BGB_GBB',
                                        'BGB_BBG', 'BGB_BGB', 'BGB_BBB',
                                        'BBB_GGG', 'BBB_BGG', 'BBB_GBG', 'BBB_GGB', 'BBB_GBB',
                                        'BBB_BBG', 'BBB_BGB', 'BBB_BBB'

                                        ])

    print(weather_chain.next_state(current_state='GGG_GGG'))

    print(weather_chain.next_state(current_state='Snowy'))

    weather_chain.generate_states(current_state='Snowy', no=10)


    #env = MarkovChain()

    #RL = QLearningTable(actions=list(range(env.n_actions)))
    #update()
    #env.after(100, update)
    #env.mainloop()
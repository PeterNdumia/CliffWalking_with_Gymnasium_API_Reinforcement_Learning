import numpy as np
import time
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import gymnasium
import gymnasium_env
env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human")
observation, info = env.reset()

# QTable : contains the Q-Values for every (state,action) pair

numstates=(env.observation_space['agent'].high[0]+1) * (env.observation_space['agent'].high[1]+1)
numactions = env.action_space.n
qtable = np.random.rand(numstates, numactions).tolist()

# hyperparameters
#number of episodes
episodes = 50
#discount factor
gamma = 0.1
# probability of randomness for the next action
epsilon = 0.08
# decay rate of the epsilon
decay = 0.1
#learning rate
alpha =1


rewards_per_episode = []
steps_per_episode = []

for i in range(episodes):
    state_dict, info = env.reset()
    state = (state_dict['agent'][0]+1) * (state_dict['agent'][1]+1) - 1
    steps = 0
    # total reward initialized to 0
    total_reward = 0
    terminated = False

    while not terminated:
        os.system('clear')
        print("episode #", i+1, "/", episodes)
        env.render()
        time.sleep(0.05)

        # count steps to finish game
        steps += 1
        steps_per_episode.append(steps)

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        # if not select max action in Qtable (act greedy)
        else:
            action = qtable[state].index(max(qtable[state]))

        
        
        # take action
        next_state_dict, reward, terminated, fell_off_cliff, info = env.step(action)
        next_state = (next_state_dict['agent'][0]+1)* (next_state_dict['agent'][1]+1) - 1

        # update qtable value with Bellman equation
        qtable[state][action] =  qtable[state][action] + alpha*(reward + gamma * max(qtable[next_state])-qtable[state][action])


        # update state
        state = next_state

        #total reward
        total_reward += reward
        rewards_per_episode.append(total_reward)

    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon
    #print number of steps
    print("\nDone in", steps, "steps".format(steps))
    #print total reward
    print("Total reward :", total_reward)
    time.sleep(0.8)

fig, axes  = plt.subplots(2,figsize=(10, 6))
axes[0].plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode, label="Total Reward per Episode", color="blue", marker="o")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].set_title("Total Reward per Episode Over Time")
axes[0].legend()
axes[0].grid(True)

axes[1].figure(figsize=(10, 6))
axes[1].plot(range(1, len(steps_per_episode) + 1), steps_per_episode, label="Total steps per Episode", color="blue", marker="o")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Total Steps")
axes[1].set_title("Total Steps per Episode Over Time")
axes[1].legend()
axes[1].grid(True)
axes[1].savefig("Rewards and steps per episode.png")
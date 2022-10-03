# Local module imports
import numpy as np
import random
from agent_class import Game_Agent
from env_gym import CartPole_Env

gym_env = CartPole_Env()
agent = Game_Agent(gym_env)
agent.load_model()

game_score = []
score_ave = 0

# Loop through episodes
print("Testing model")
for e in range(1, 10):

        # Set the initial state for the episode
        state = agent.env_class.reset_env()

        # Variables to track the process
        done = False # If episode over
        i = 0 # Time step counter

        # Training loop for time steps inside an episode
        while not done:
            agent.env.render()
            state = np.reshape(state, (1, 4))
            action = agent.q_eval(state)

            action = np.argmax(action)

            # Takes step based on action
            next_state, reward, done, extra = agent.env_class.env_action(action)

            # Itterate counters
            i += 1

            # Reward structure
            if not done:
                reward = reward

            # Reward when game over
            else:
                if i == 500:
                    reward = 1000
                else:
                    reward = -1000

            # Move to the next state
            state = next_state

            # End of episode
            if done:
                game_score.append(i)                
                print(f"Episode: {e}/10 - Score: {i}")
     

buffer_ave = sum(game_score)/len(game_score)
print("Average score over 10 rounds: ", buffer_ave)
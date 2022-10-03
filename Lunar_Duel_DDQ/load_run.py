# Local module imports
import numpy as np
import random
from agent_class import Game_Agent
from env_gym import Lunar_Env

# Load environment
gym_env = Lunar_Env()

# Load agent
agent = Game_Agent(gym_env)
agent.load_model()

game_score = []
score_ave = 0

# Loop through episodes
print("Testing model")
for e in range(1, 11):

        # Set the initial state for the episode
        state = agent.env_class.reset_env()

        # Variables to track the process
        done = False # If episode over
        i = 0 # Time step counter
        score = 0

        # Training loop for time steps inside an episode
        while not done:
            agent.env.render()
            
            state = np.reshape(state, (1, 8))
            action = agent.q_eval(state)
            action = np.argmax(action)
            #action = random.randint(0, 1)
            # Takes step based on action
            next_state, reward, done, extra = agent.env_class.env_action(action)

            # Itterate counters
            i += 1

            # Reward structure
            if not done:
                reward = reward

            score += reward

            # Move to the next state
            state = next_state

            # End of episode
            if done:
                game_score.append(score)                
                print(f"Episode: {e}/10 - Score: {score}")
     

buffer_ave = sum(game_score)/len(game_score)
print("Average score over 10 rounds: ", buffer_ave)
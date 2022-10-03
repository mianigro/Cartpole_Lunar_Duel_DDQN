# Local module imports
import numpy as np
import os
from agent_class import Game_Agent
from env_gym import Lunar_Env


gym_env = Lunar_Env()
agent = Game_Agent(gym_env)

# Load previouse model if needed
if agent.load_model_status == True:
    agent.load_model()

# Stops old logs accumulating
if os.path.exists("log_file.csv"):
        print("Deleting log file.")
        os.remove('log_file.csv')


# Tracking game count for accumulating memory
game_count = 0

# Build memory bank
print("Building memory bank")
while agent.train_start_counter < agent.train_start:

        state = agent.env_class.reset_env()
        done = False

        # Memory accumulation loop
        while not done:
            #agent.env.render()


            action = agent.do_action(state)
            next_state, reward, done, extra = agent.env_class.env_action(action)

            # Reward structure
            if not done:
                reward = reward

            agent.to_memory(state, action, reward, next_state, done)

            # Itterating to next state
            state = next_state

            # End of episode
            if done:             
                print(f"Game Count: {game_count} \
                    - Memory Count: {agent.train_start_counter}")
            
            agent.train_start_counter += 1

        game_count += 1


# Training game scoring
game_score = []
buffer_ave = -99999999

# Training model loop
print("Training model")
for e in range(agent.EPISODES):

        state = agent.env_class.reset_env()
        agent.episode_count = e
        done = False
        score = 0

        # Game loop inside an episode
        while not done:
            if agent.view_render == True:
                agent.env.render() 
                
            action = agent.do_action(state)
            next_state, reward, done, extra = agent.env_class.env_action(action)

            # Reward structure
            if not done:
                reward = reward

            score += reward
            agent.it_count += 1

            agent.to_memory(state, action, reward, next_state, done)

            # Itterating to next state
            state = next_state

            # End of episode
            if done:
                game_score.append(score)            
                print(f"Episode: {e}/{agent.EPISODES} \
                    - Epsilon: {round(agent.epsilon, 5)} - Score: {score}")
                agent.log_info(e, agent.epsilon, score)

            agent.replay_train()
            
        # Track best 10 round average
        if len(game_score) == 10:
            print(f"Average last 10 rounds: {sum(game_score)/len(game_score)}")
            print(f"- Prev Best: {buffer_ave}")
            
            if sum(game_score)/len(game_score) > buffer_ave:
                buffer_ave = sum(game_score)/len(game_score)
            
            game_score = []

        # Save model if needed with best score
        if agent.save_model_status == True and \
            agent.best_score_saved < buffer_ave:
            if agent.episode_count % agent.episodes_to_save == 0:
                agent.save()
                agent.best_score_saved = buffer_ave
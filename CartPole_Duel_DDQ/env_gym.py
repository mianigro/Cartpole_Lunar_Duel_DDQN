# Third party imports
import gym
import numpy as np

class CartPole_Env:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_size = self.env.observation_space.shape[0] # Shape of state array
        self.action_space = 2
        print(self.state_size)
        print(self.action_space)

    def reset_env(self):
        state = self.env.reset()
        return state


    def env_action(self, action):
        next_state, reward, done, extra = self.env.step(action)
        
        return next_state, reward, done, extra

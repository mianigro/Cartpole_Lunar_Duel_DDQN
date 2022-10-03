from collections import deque
import numpy as np
import random

class Replay_Buffer():

    def __init__(self, state_size, mem_size, batch_size):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.memory_count = 0
        self.state_size = state_size
        self.state_mem = np.zeros((self.mem_size, state_size))
        self.new_state_mem = np.zeros((self.mem_size, state_size))
        self.action_mem = np.zeros(self.mem_size)
        self.reward_mem = np.zeros(self.mem_size)
        self.done_mem = np.zeros(self.mem_size)

    
    def store_memory(self, state, action, reward, next_state, done):
        # Allows memory to restart as memory counter is > mem size
        index = self.memory_count % self.mem_size
        
        self.state_mem[index] = state
        self.new_state_mem[index] = next_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem[index] = done

        self.memory_count += 1 # Itterate memory counter


    def len_buffer(self):
        return self.memory_count

    
    def sample_buffer(self):
        # Allow a good sample before training
        if self.memory_count < self.batch_size:
            return

        mem_amount = min(self.memory_count, self.mem_size)
        batch_mem = np.random.choice(
            mem_amount, 
            self.batch_size, 
            replace=False)
            
        state = self.state_mem[batch_mem]
        next_state = self.new_state_mem[batch_mem]
        action = self.action_mem[batch_mem]
        reward = self.reward_mem[batch_mem]
        done = self.done_mem[batch_mem]

        return state, action, reward, next_state, done
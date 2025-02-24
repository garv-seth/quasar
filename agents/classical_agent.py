import numpy as np
from collections import deque
import random

class ClassicalAgent:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple neural network weights
        self.weights = np.random.randn(state_dim, action_dim) / np.sqrt(state_dim)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        action_values = np.dot(state, self.weights)
        return np.argmax(action_values)

    def train_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.memory.append((state, action, reward, next_state, done))
            
            # Train on batch
            if len(self.memory) >= 32:
                self._train_batch()
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward

    def _train_batch(self):
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Simple policy gradient update
        gradients = np.zeros_like(self.weights)
        for i in range(len(batch)):
            target = rewards[i]
            if not dones[i]:
                target += 0.99 * np.max(np.dot(next_states[i], self.weights))
            
            current = np.dot(states[i], self.weights)
            gradients += states[i].reshape(-1, 1) * (target - current)
        
        self.weights += self.learning_rate * gradients / len(batch)

import numpy as np
from collections import deque
import random

class HybridMemory:
    def __init__(self, capacity):
        self.classical_memory = deque(maxlen=capacity)
        self.quantum_state_cache = {}  # Simulated quantum state cache
        self.capacity = capacity
    
    def store(self, state, action, reward, next_state, done):
        # Store in classical memory
        self.classical_memory.append((state, action, reward, next_state, done))
        
        # Simulate quantum state caching
        state_hash = hash(state.tobytes())
        self.quantum_state_cache[state_hash] = {
            'state_vector': state,
            'frequency': self.quantum_state_cache.get(state_hash, {'frequency': 0})['frequency'] + 1
        }
        
        # Clear old cache entries if needed
        if len(self.quantum_state_cache) > self.capacity:
            min_freq_key = min(self.quantum_state_cache, 
                             key=lambda k: self.quantum_state_cache[k]['frequency'])
            del self.quantum_state_cache[min_freq_key]
    
    def sample(self, batch_size):
        return random.sample(self.classical_memory, batch_size)
    
    def __len__(self):
        return len(self.classical_memory)

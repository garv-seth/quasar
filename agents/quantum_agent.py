import pennylane as qml
import numpy as np
from utils.quantum_circuits import create_adaptive_circuit
from utils.memory import HybridMemory

class QUASARAgent:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.memory = HybridMemory(capacity=10000)
        
        # Initialize quantum device
        self.n_qubits = state_dim + 2  # Additional qubits for entanglement
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Initialize quantum circuit parameters
        self.params = np.random.uniform(-np.pi, np.pi, (3, self.n_qubits, 3))
        
        # Create quantum circuit
        self.circuit = qml.QNode(create_adaptive_circuit, self.dev)

    def get_action(self, state):
        # Normalize state
        normalized_state = state / np.linalg.norm(state)
        
        # Get quantum circuit output
        circuit_output = self.circuit(self.params, normalized_state)
        
        # Apply quantum-inspired action selection
        action_probs = np.abs(circuit_output)**2
        action = np.random.choice(self.action_dim, p=action_probs)
        
        return action

    def train_episode(self, env):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in hybrid memory
            self.memory.store(state, action, reward, next_state, done)
            
            # Update quantum parameters using policy gradient
            if len(self.memory) >= 32:
                self._update_parameters()
            
            state = next_state
            total_reward += reward
        
        return total_reward

    def _update_parameters(self):
        batch = self.memory.sample(32)
        states, actions, rewards, _, _ = zip(*batch)
        
        # Compute quantum gradients
        gradients = qml.grad(self.circuit)(self.params, np.mean(states, axis=0))
        
        # Update parameters using policy gradient
        self.params -= self.learning_rate * gradients * np.mean(rewards)

    def get_circuit_params(self):
        return self.params

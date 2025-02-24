import numpy as np
import pandas as pd

def calculate_metrics(quantum_metrics, classical_metrics):
    metrics = {
        'Metric': [
            'Average Reward',
            'Max Reward',
            'Convergence Episodes',
            'Final Performance',
            'Learning Stability'
        ],
        'QUASAR Agent': [
            np.mean(quantum_metrics),
            np.max(quantum_metrics),
            _calculate_convergence(quantum_metrics),
            np.mean(quantum_metrics[-10:]),
            np.std(quantum_metrics)
        ],
        'Classical Agent': [
            np.mean(classical_metrics),
            np.max(classical_metrics),
            _calculate_convergence(classical_metrics),
            np.mean(classical_metrics[-10:]),
            np.std(classical_metrics)
        ]
    }
    
    return pd.DataFrame(metrics).set_index('Metric')

def _calculate_convergence(rewards, threshold=195):
    """Calculate number of episodes until convergence"""
    for i, reward in enumerate(rewards):
        if reward >= threshold:
            return i + 1
    return len(rewards)

import plotly.graph_objects as go
import numpy as np

def create_comparison_chart(quantum_metrics, classical_metrics):
    fig = go.Figure()
    
    # Add quantum agent performance
    fig.add_trace(go.Scatter(
        y=quantum_metrics,
        name='QUASAR Agent',
        line=dict(color='#7b2cbf', width=2),
        mode='lines+markers'
    ))
    
    # Add classical agent performance
    fig.add_trace(go.Scatter(
        y=classical_metrics,
        name='Classical Agent',
        line=dict(color='#2cb5e8', width=2),
        mode='lines+markers'
    ))
    
    # Update layout
    fig.update_layout(
        title='Agent Performance Comparison',
        xaxis_title='Episode',
        yaxis_title='Reward',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_circuit_visualization(circuit_params):
    # Create a visual representation of the quantum circuit
    fig = go.Figure()
    
    n_layers, n_qubits, n_params = circuit_params.shape
    
    # Create heatmap of circuit parameters
    fig.add_trace(go.Heatmap(
        z=np.mean(circuit_params, axis=2),
        x=[f'Qubit {i}' for i in range(n_qubits)],
        y=[f'Layer {i}' for i in range(n_layers)],
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title='Quantum Circuit Parameters',
        xaxis_title='Qubits',
        yaxis_title='Circuit Layers',
        template='plotly_dark'
    )
    
    return fig

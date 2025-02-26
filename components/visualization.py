"""Enhanced visualization utilities for the QUASAR framework."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
import pennylane as qml


def create_comparison_chart(quantum_metrics, classical_metrics):
    """
    Create comparison chart between quantum and classical performance.

    Args:
        quantum_metrics: List of quantum performance metrics
        classical_metrics: List of classical performance metrics

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add quantum agent performance
    fig.add_trace(
        go.Scatter(y=quantum_metrics,
                   name='QUASAR Agent',
                   line=dict(color='#7b2cbf', width=2),
                   mode='lines+markers'))

    # Add classical agent performance
    fig.add_trace(
        go.Scatter(y=classical_metrics,
                   name='Classical Agent',
                   line=dict(color='#2cb5e8', width=2),
                   mode='lines+markers'))

    # Update layout
    fig.update_layout(title='Agent Performance Comparison',
                      xaxis_title='Episode',
                      yaxis_title='Reward',
                      template='plotly_dark',
                      hovermode='x unified',
                      legend=dict(yanchor="top",
                                  y=0.99,
                                  xanchor="left",
                                  x=0.01))

    return fig


def create_circuit_visualization(circuit_params):
    """
    Create a visual representation of quantum circuit parameters.

    Args:
        circuit_params: Circuit parameters array

    Returns:
        Plotly figure object
    """
    # Create a visual representation of the quantum circuit
    fig = go.Figure()

    if len(circuit_params.shape) == 3:
        n_layers, n_qubits, n_params = circuit_params.shape

        # Create heatmap of circuit parameters
        fig.add_trace(
            go.Heatmap(z=np.mean(circuit_params, axis=2),
                       x=[f'Qubit {i}' for i in range(n_qubits)],
                       y=[f'Layer {i}' for i in range(n_layers)],
                       colorscale='Viridis',
                       showscale=True))

        fig.update_layout(title='Quantum Circuit Parameters',
                          xaxis_title='Qubits',
                          yaxis_title='Circuit Layers',
                          template='plotly_dark')
    else:
        # Fallback for other parameter shapes
        fig.add_trace(
            go.Heatmap(z=circuit_params, colorscale='Viridis', showscale=True))

        fig.update_layout(title='Quantum Circuit Parameters',
                          template='plotly_dark')

    return fig


def create_factorization_tree(factors: List[int], number: int):
    """
    Create a factor tree visualization.

    Args:
        factors: List of factors
        number: Number being factorized

    Returns:
        Plotly figure object
    """
    # Create nodes and edges for the factor tree
    nodes = []
    edges = []

    # Add the original number as the root node
    nodes.append({
        'id': str(number),
        'label': str(number),
        'level': 0,
        'size': 20
    })

    # Add factor pairs
    level = 1
    to_process = [number]
    processed = set()

    while to_process:
        current = to_process.pop(0)
        if current in processed or current == 1:
            continue

        processed.add(current)

        # Find factors of current number
        current_factors = [
            f for f in factors if current % f == 0 and f != 1 and f != current
        ]

        if not current_factors:
            continue

        # Add at most two factors per level for clarity
        if len(current_factors) >= 2:
            f1, f2 = current_factors[0], current // current_factors[0]

            nodes.append({
                'id': f"{current}_{f1}",
                'label': str(f1),
                'level': level,
                'size': 15
            })

            nodes.append({
                'id': f"{current}_{f2}",
                'label': str(f2),
                'level': level,
                'size': 15
            })

            edges.append({
                'source': str(current),
                'target': f"{current}_{f1}",
                'label': '×'
            })

            edges.append({
                'source': str(current),
                'target': f"{current}_{f2}",
                'label': '×'
            })

            # Add to processing queue
            if f1 not in processed and f1 != 1:
                to_process.append(f1)
            if f2 not in processed and f2 != 1:
                to_process.append(f2)

        level += 1

    # Create a simplified version for Plotly
    # First, lay out nodes in a tree structure
    node_positions = {}
    level_counts = {}

    for node in nodes:
        level = node['level']
        level_counts[level] = level_counts.get(level, 0) + 1

    for node in nodes:
        level = node['level']
        level_count = level_counts[level]
        idx = len(
            [n for n in nodes if n['level'] == level and n['id'] < node['id']])

        # Calculate x position based on level index
        x = idx / max(1, level_count - 1) if level_count > 1 else 0.5
        y = -level / max(1,
                         len(level_counts) - 1) if len(level_counts) > 1 else 0

        node_positions[node['id']] = (x, y)

    # Create the network visualization
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in nodes:
        x, y = node_positions[node['id']]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node['label'])
        node_size.append(node['size'])

    # Create edge traces
    edge_x = []
    edge_y = []

    for edge in edges:
        x0, y0 = node_positions[edge['source']]
        x1, y1 = node_positions[edge['target']]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(
        go.Scatter(x=edge_x,
                   y=edge_y,
                   mode='lines',
                   line=dict(width=1, color='#888'),
                   hoverinfo='none'))

    # Add nodes
    fig.add_trace(
        go.Scatter(x=node_x,
                   y=node_y,
                   mode='markers+text',
                   text=node_text,
                   marker=dict(showscale=False,
                               color='#7b2cbf',
                               size=node_size,
                               line=dict(width=2, color='#fff')),
                   textposition="middle center",
                   textfont=dict(color='white')))

    # Update layout
    fig.update_layout(title=f"Factor Tree for {number}",
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      yaxis=dict(showgrid=False,
                                 zeroline=False,
                                 showticklabels=False),
                      template='plotly_dark')

    return fig


def create_3d_quantum_state(state_vector: np.ndarray):
    """
    Create a 3D visualization of a quantum state.

    Args:
        state_vector: Quantum state vector

    Returns:
        Plotly figure object
    """
    # Convert to Bloch sphere coordinates
    if len(state_vector) >= 2:
        # For a qubit state (assume first two amplitudes represent |0⟩ and |1⟩)
        alpha = state_vector[0]
        beta = state_vector[1] if len(state_vector) > 1 else 0

        # Normalize if needed
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        if norm > 0:
            alpha = alpha / norm
            beta = beta / norm

        # Convert to Bloch sphere coordinates
        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta) - np.angle(alpha)

        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Create Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))

        # Create figure
        fig = go.Figure()

        # Add Bloch sphere (transparent)
        fig.add_trace(
            go.Surface(x=sphere_x,
                       y=sphere_y,
                       z=sphere_z,
                       opacity=0.2,
                       colorscale='Blues',
                       showscale=False))

        # Add axes
        axis_length = 1.2

        # X axis (red)
        fig.add_trace(
            go.Scatter3d(x=[-axis_length, axis_length],
                         y=[0, 0],
                         z=[0, 0],
                         mode='lines',
                         line=dict(color='red', width=3),
                         showlegend=False))

        # Y axis (green)
        fig.add_trace(
            go.Scatter3d(x=[0, 0],
                         y=[-axis_length, axis_length],
                         z=[0, 0],
                         mode='lines',
                         line=dict(color='green', width=3),
                         showlegend=False))

        # Z axis (blue)
        fig.add_trace(
            go.Scatter3d(x=[0, 0],
                         y=[0, 0],
                         z=[-axis_length, axis_length],
                         mode='lines',
                         line=dict(color='blue', width=3),
                         showlegend=False))

        # Add state vector
        fig.add_trace(
            go.Scatter3d(x=[0, x],
                         y=[0, y],
                         z=[0, z],
                         mode='lines+markers',
                         line=dict(color='purple', width=5),
                         marker=dict(size=7, color='purple'),
                         showlegend=False))

        # Update layout
        fig.update_layout(title='Quantum State Visualization (Bloch Sphere)',
                          scene=dict(xaxis_title="X",
                                     yaxis_title="Y",
                                     zaxis_title="Z",
                                     aspectmode='cube'),
                          template='plotly_dark')

    else:
        # Fallback for invalid state vector
        fig = go.Figure()
        fig.add_annotation(
            text="Invalid quantum state vector for visualization",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14))

        fig.update_layout(title='Quantum State Visualization (Error)',
                          template='plotly_dark')

    return fig


def create_execution_time_comparison(quantum_times: List[float],
                                     classical_times: List[float],
                                     labels: List[str]):
    """
    Create a comparison of execution times.

    Args:
        quantum_times: List of quantum execution times
        classical_times: List of classical execution times
        labels: List of labels for each comparison

    Returns:
        Plotly figure object
    """
    # Calculate speedups
    speedups = [
        c / q if q > 0 else 0 for q, c in zip(quantum_times, classical_times)
    ]

    # Create figure
    fig = go.Figure()

    # Add bars for quantum times
    fig.add_trace(
        go.Bar(x=labels,
               y=quantum_times,
               name='Quantum Execution Time',
               marker_color='#7b2cbf'))

    # Add bars for classical times
    fig.add_trace(
        go.Bar(x=labels,
               y=classical_times,
               name='Classical Execution Time',
               marker_color='#2cb5e8'))

    # Add speedup indicators
    for i, (label, speedup) in enumerate(zip(labels, speedups)):
        if speedup > 0:
            fig.add_annotation(x=label,
                               y=max(quantum_times[i], classical_times[i]) *
                               1.1,
                               text=f"{speedup:.2f}x speedup",
                               showarrow=False,
                               font=dict(color='#ff9800', size=12))

    # Update layout
    fig.update_layout(title='Execution Time Comparison',
                      xaxis_title='Task',
                      yaxis_title='Time (seconds)',
                      barmode='group',
                      template='plotly_dark')

    return fig


def create_quantum_circuit_diagram(circuit_description: Dict[str, Any]):
    """
    Create a visual representation of a quantum circuit.

    Args:
        circuit_description: Dictionary with circuit description

    Returns:
        Plotly figure object or string (based on circuit complexity)
    """
    try:
        # Extract circuit details
        n_qubits = circuit_description.get('n_qubits', 4)
        gates = circuit_description.get('gates', [])

        # Simple circuits can be visualized directly
        if n_qubits <= 8 and len(gates) <= 20:
            # Create a PennyLane device
            dev = qml.device("default.qubit", wires=n_qubits)

            # Define a basic quantum function
            @qml.qnode(dev)
            def circuit():
                # Apply gates based on the description
                for gate in gates:
                    gate_type = gate.get('type')
                    wires = gate.get('wires', [0])
                    params = gate.get('params', [])

                    if gate_type == 'Hadamard':
                        qml.Hadamard(wires=wires[0])
                    elif gate_type == 'PauliX':
                        qml.PauliX(wires=wires[0])
                    elif gate_type == 'PauliY':
                        qml.PauliY(wires=wires[0])
                    elif gate_type == 'PauliZ':
                        qml.PauliZ(wires=wires[0])
                    elif gate_type == 'CNOT':
                        qml.CNOT(wires=wires)
                    elif gate_type == 'CZ':
                        qml.CZ(wires=wires)
                    elif gate_type == 'RX':
                        qml.RX(params[0], wires=wires[0])
                    elif gate_type == 'RY':
                        qml.RY(params[0], wires=wires[0])
                    elif gate_type == 'RZ':
                        qml.RZ(params[0], wires=wires[0])
                    elif gate_type == 'Rot':
                        qml.Rot(*params, wires=wires[0])

                return qml.state()

            # Create the circuit diagram
            fig, ax = qml.draw_mpl(circuit)()
            return fig
        else:
            # For complex circuits, return a text representation
            circuit_text = f"Quantum Circuit with {n_qubits} qubits and {len(gates)} gates\n\n"

            for i, gate in enumerate(gates[:20]):
                gate_type = gate.get('type', 'Unknown')
                wires = gate.get('wires', [0])
                params = gate.get('params', [])

                if params:
                    param_str = ", ".join([f"{p:.2f}" for p in params])
                    circuit_text += f"{i+1}: {gate_type}({param_str}) on wire(s) {wires}\n"
                else:
                    circuit_text += f"{i+1}: {gate_type} on wire(s) {wires}\n"

            if len(gates) > 20:
                circuit_text += f"\n... and {len(gates) - 20} more gates"

            return circuit_text

    except Exception as e:
        logging.error(f"Error creating circuit diagram: {str(e)}")
        return f"Error creating circuit diagram: {str(e)}"


def create_performance_dashboard(metrics: Dict[str, Any]):
    """
    Create a comprehensive performance dashboard.

    Args:
        metrics: Dictionary with performance metrics

    Returns:
        Plotly figure object
    """
    # Extract metrics
    task_types = metrics.get('task_types',
                             ['Factorization', 'Search', 'Optimization'])
    quantum_times = metrics.get('quantum_times', [0.1, 0.2, 0.3])
    classical_times = metrics.get('classical_times', [0.2, 0.4, 0.9])
    success_rates = metrics.get('success_rates', [95, 90, 85])

    # Create speedup metrics
    speedups = [
        c / q if q > 0 else 0 for q, c in zip(quantum_times, classical_times)
    ]

    # Create figure with subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=('Execution Time Comparison',
                                        'Quantum Speedup', 'Success Rate',
                                        'Resource Usage'))

    # Add execution time bars
    fig.add_trace(go.Bar(x=task_types,
                         y=quantum_times,
                         name='Quantum',
                         marker_color='#7b2cbf'),
                  row=1,
                  col=1)

    fig.add_trace(go.Bar(x=task_types,
                         y=classical_times,
                         name='Classical',
                         marker_color='#2cb5e8'),
                  row=1,
                  col=1)

    # Add speedup bars
    fig.add_trace(go.Bar(
        x=task_types,
        y=speedups,
        marker_color=['#ff9800' if s > 1 else '#757575' for s in speedups],
        text=[f"{s:.2f}x" for s in speedups],
        textposition='auto',
        name='Speedup'),
                  row=1,
                  col=2)

    # Add success rate gauge
    for i, (task, rate) in enumerate(zip(task_types, success_rates)):
        fig.add_trace(go.Indicator(
            mode='gauge+number',
            value=rate,
            title={'text': task},
            gauge={
                'axis': {
                    'range': [0, 100]
                },
                'bar': {
                    'color':
                    '#7b2cbf'
                    if rate >= 90 else '#ff9800' if rate >= 70 else '#e53935'
                },
                'steps': [{
                    'range': [0, 70],
                    'color': 'rgba(255, 0, 0, 0.2)'
                }, {
                    'range': [70, 90],
                    'color': 'rgba(255, 153, 0, 0.2)'
                }, {
                    'range': [90, 100],
                    'color': 'rgba(123, 44, 191, 0.2)'
                }]
            },
            domain={
                'row': 1,
                'column': 0 if i == 0 else 1
            }),
                      row=2,
                      col=1 if i == 0 else 2)

    # Update layout
    fig.update_layout(title='Quantum Performance Dashboard',
                      template='plotly_dark',
                      height=800,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1))

    return fig


def create_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame with performance metrics.

    Args:
        metrics: Dictionary with performance metrics

    Returns:
        pandas DataFrame with metrics
    """
    # Extract task types and metrics
    task_types = metrics.get('task_types',
                             ['Factorization', 'Search', 'Optimization'])
    quantum_times = metrics.get('quantum_times', [0.1, 0.2, 0.3])
    classical_times = metrics.get('classical_times', [0.2, 0.4, 0.9])
    speedups = [
        c / q if q > 0 else 0 for q, c in zip(quantum_times, classical_times)
    ]

    # Additional metrics
    theoretical_speedups = metrics.get(
        'theoretical_speedups', ['Exponential', 'Quadratic', 'Polynomial'])

    best_use_cases = metrics.get('best_use_cases', [
        'Large number factorization', 'Unstructured database search',
        'Resource allocation problems'
    ])

    # Create DataFrame
    df = pd.DataFrame({
        'Task Type':
        task_types,
        'Quantum Time (s)': [f"{t:.4f}" for t in quantum_times],
        'Classical Time (s)': [f"{t:.4f}" for t in classical_times],
        'Speedup': [f"{s:.2f}x" for s in speedups],
        'Theoretical Speedup':
        theoretical_speedups,
        'Best Use Case':
        best_use_cases
    })

    return df

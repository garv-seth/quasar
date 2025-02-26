            """Integration layer for hybrid classical-quantum computations in QUASAR framework."""

            import numpy as np
            from typing import Any, Dict, List, Optional, Union
            import json
            import logging
            import aiohttp
            import asyncio
            from datetime import datetime
            import xml.etree.ElementTree as ET
            import re

            from ..quantum.optimizer import QuantumOptimizer
            from ..quantum.factorization_manager import FactorizationManager
            from ..classical import Memory, Retriever
            from openai import AsyncOpenAI

            class HybridComputation:
                """Manages hybrid classical-quantum computations with enhanced source integration."""

                def __init__(self, 
                             n_qubits: int = 8,
                             use_quantum: bool = True,
                             memory_size: int = 100,
                             use_azure: bool = True):
                    """Initialize hybrid computation manager."""
                    self.use_quantum = use_quantum
                    self.n_qubits = min(n_qubits, 29)  # Maximum qubits for IonQ
                    self.memory = Memory(max_messages=memory_size)
                    self.retriever = Retriever()
                    self.openai_client = AsyncOpenAI()

                    # Thresholds for using quantum computing
                    self.factorization_threshold = 100000  # Numbers above this use quantum methods
                    self.search_threshold = 1000  # Databases larger than this use quantum search
                    self.optimization_threshold = 10  # Problems with more than these variables use quantum

                    # Academic and research API endpoints
                    self.api_endpoints = {
                        "arxiv": "http://export.arxiv.org/api/query",
                        "quantum_papers": "https://quantum-computing.ibm.com/api/papers",
                        "ionq_docs": "https://ionq.com/docs/api"
                    }

                    if use_quantum:
                        try:
                            # Initialize quantum optimizer with Azure backend
                            self.quantum_optimizer = QuantumOptimizer(
                                n_qubits=self.n_qubits,
                                use_azure=use_azure
                            )
                            # Initialize factorization manager with the optimizer
                            self.factorization_manager = FactorizationManager(self.quantum_optimizer)
                            logging.info(f"Successfully initialized quantum components with {self.n_qubits} qubits")
                        except Exception as e:
                            logging.error(f"Failed to initialize quantum components: {str(e)}")
                            self.use_quantum = False
                            logging.info("Falling back to classical processing")

                async def process_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                    """Process a task using hybrid computation with clear decision criteria."""
                    try:
                        start_time = datetime.now()

                        # First, classify the task to determine the best processing method
                        task_classification = await self._classify_task(task)
                        task_type = task_classification["type"]

                        # Initialize results dictionary
                        results = {
                            'task': task,
                            'timestamp': datetime.now().isoformat(),
                            'task_type': task_type,
                            'processing_method': None,
                            'computation_time': None
                        }

                        # Process based on task type
                        if task_type == "factorization":
                            number = task_classification.get("number", 0)
                            if number > 0:
                                # Determine if we should use quantum or classical methods
                                use_quantum_for_task = (
                                    self.use_quantum and 
                                    number > self.factorization_threshold and 
                                    hasattr(self, 'factorization_manager')
                                )

                                results['processing_method'] = "quantum" if use_quantum_for_task else "classical"

                                # Process factorization
                                factorization_result = await self.factorization_manager.factorize(number)

                                # Add results
                                results.update({
                                    'method_used': factorization_result.method_used,
                                    'backend': factorization_result.details.get('backend', 'Unknown'),
                                    'computation_time': factorization_result.computation_time,
                                    'factors': factorization_result.factors,
                                    'details': factorization_result.details,
                                    'success': factorization_result.success
                                })

                                # Generate explanation using GPT-4o
                                explanation = await self._generate_explanation(
                                    task_type="factorization",
                                    number=number,
                                    factors=factorization_result.factors,
                                    method=factorization_result.method_used,
                                    computation_time=factorization_result.computation_time
                                )

                                results['response'] = explanation

                        elif task_type == "optimization":
                            # Extract optimization parameters
                            optimization_params = task_classification.get("parameters", {})
                            problem_size = len(optimization_params.get("variables", []))

                            # Determine if we should use quantum methods
                            use_quantum_for_task = (
                                self.use_quantum and 
                                problem_size > self.optimization_threshold and
                                hasattr(self, 'quantum_optimizer')
                            )

                            results['processing_method'] = "quantum" if use_quantum_for_task else "classical"

                            if use_quantum_for_task:
                                # Process with quantum optimization (QAOA)
                                optimization_result = self.quantum_optimizer.optimize_resource_allocation(optimization_params)
                                results.update(optimization_result)
                            else:
                                # Use classical optimization
                                optimization_result = await self._classical_optimization(optimization_params)
                                results.update(optimization_result)

                            # Generate explanation
                            explanation = await self._generate_explanation(
                                task_type="optimization",
                                parameters=optimization_params,
                                result=optimization_result,
                                method=results['processing_method']
                            )

                            results['response'] = explanation

                        elif task_type == "search":
                            # Extract search parameters
                            search_params = task_classification.get("parameters", {})
                            database_size = search_params.get("database_size", 0)

                            # Determine if we should use quantum methods
                            use_quantum_for_task = (
                                self.use_quantum and 
                                database_size > self.search_threshold and
                                hasattr(self, 'quantum_optimizer')
                            )

                            results['processing_method'] = "quantum" if use_quantum_for_task else "classical"

                            if use_quantum_for_task:
                                # Process with quantum search (Grover's algorithm)
                                search_result = await self._quantum_search(search_params)
                                results.update(search_result)
                            else:
                                # Use classical search
                                search_result = await self._classical_search(search_params)
                                results.update(search_result)

                            # Generate explanation
                            explanation = await self._generate_explanation(
                                task_type="search",
                                parameters=search_params,
                                result=search_result,
                                method=results['processing_method']
                            )

                            results['response'] = explanation

                        else:
                            # Process general queries with classical methods (GPT-4o)
                            classical_result = await self._classical_process(task)
                            results.update({
                                'processing_method': 'classical',
                                'classical_result': classical_result,
                                'response': classical_result.get('response', 'I processed your request using classical methods.'),
                                'computation_time': (datetime.now() - start_time).total_seconds()
                            })

                        # Update memory with the results
                        self._update_memory(task, results)

                        # Include quantum metrics if quantum methods were used
                        if results.get('processing_method') == 'quantum' and hasattr(self, 'quantum_optimizer'):
                            results['quantum_metrics'] = self.get_quantum_metrics()

                        return results

                    except Exception as e:
                        error_msg = f"Error processing task: {str(e)}"
                        logging.error(error_msg)
                        return {
                            'error': True,
                            'message': error_msg,
                            'task': task,
                            'task_type': 'error'
                        }

                async def _classify_task(self, task: str) -> Dict[str, Any]:
                    """Classify the task to determine the appropriate processing method."""
                    # First, check for factorization tasks
                    factorization_match = re.search(r'factor.*?(\d+)', task.lower())
                    if factorization_match:
                        number = int(factorization_match.group(1))
                        return {"type": "factorization", "number": number}

                    # Check for optimization keywords
                    if any(word in task.lower() for word in ['optimize', 'optimization', 'minimum', 'maximum', 'resource']):
                        # Extract parameters for optimization task
                        return {
                            "type": "optimization",
                            "parameters": await self._extract_optimization_parameters(task)
                        }

                    # Check for search keywords
                    if any(word in task.lower() for word in ['search', 'find', 'locate', 'query']):
                        return {
                            "type": "search",
                            "parameters": await self._extract_search_parameters(task)
                        }

                    # Use GPT-4o for more sophisticated task classification
                    try:
                        messages = [
                            {"role": "system", "content": """You are a quantum computing task analyzer.
                            Analyze the task and categorize it as one of:
                            1. "factorization" - For factoring numbers, finding divisors, etc.
                            2. "optimization" - For resource allocation, minimization/maximization
                            3. "search" - For database or content search queries
                            4. "classical" - For general questions or tasks not suited for quantum computing

                            Return JSON in format: {"type": "[category]", "parameters": {...}}
                            For factorization, include a "number" parameter.
                            For optimization, include relevant parameters like "constraints", "objective".
                            For search, include parameters like "query", "database_size".
                            """},
                            {"role": "user", "content": task}
                        ]

                        completion = await self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=150,
                            temperature=0,
                            response_format={"type": "json_object"}
                        )

                        response = completion.choices[0].message.content
                        return json.loads(response)
                    except Exception as e:
                        logging.error(f"Task classification error: {str(e)}")
                        return {"type": "classical", "parameters": {}}

                async def _extract_optimization_parameters(self, task: str) -> Dict[str, Any]:
                    """Extract optimization parameters from the task."""
                    try:
                        messages = [
                            {"role": "system", "content": """You are an expert in optimization problems.
                            Extract optimization parameters from the task description.
                            Return JSON with:
                            - "objective_type": "minimize" or "maximize"
                            - "variables": list of variables to optimize
                            - "constraints": list of constraints
                            - "resource_count": approximate number of resources/items to optimize
                            """},
                            {"role": "user", "content": task}
                        ]

                        completion = await self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=200,
                            temperature=0,
                            response_format={"type": "json_object"}
                        )

                        response = completion.choices[0].message.content
                        return json.loads(response)
                    except Exception as e:
                        logging.error(f"Optimization parameter extraction error: {str(e)}")
                        return {
                            "objective_type": "unknown",
                            "variables": [],
                            "constraints": [],
                            "resource_count": 10  # Default value
                        }

                async def _extract_search_parameters(self, task: str) -> Dict[str, Any]:
                    """Extract search parameters from the task."""
                    try:
                        messages = [
                            {"role": "system", "content": """You are an expert in search algorithms.
                            Extract search parameters from the task description.
                            Return JSON with:
                            - "query": the search query
                            - "database_size": estimated size of database to search (default: 1000)
                            - "search_type": "exact" or "approximate"
                            """},
                            {"role": "user", "content": task}
                        ]

                        completion = await self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=200,
                            temperature=0,
                            response_format={"type": "json_object"}
                        )

                        response = completion.choices[0].message.content
                        return json.loads(response)
                    except Exception as e:
                        logging.error(f"Search parameter extraction error: {str(e)}")
                        return {
                            "query": task,
                            "database_size": 1000,  # Default value
                            "search_type": "exact"
                        }

                async def _classical_process(self, task: str) -> Dict[str, Any]:
                    """Process a task using classical methods (GPT-4o)."""
                    try:
                        messages = [
                            {"role": "system", "content": """You are a helpful assistant with expertise in quantum computing.
                            Answer the user's question clearly and concisely.
                            If the question is about quantum computing, provide accurate information.
                            """},
                            {"role": "user", "content": task}
                        ]

                        completion = await self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=500,
                            temperature=0.7
                        )

                        response = completion.choices[0].message.content
                        return {
                            "processed_result": "Task processed using classical AI",
                            "reasoning": "Task determined to be more efficient with classical computation",
                            "response": response
                        }
                    except Exception as e:
                        logging.error(f"Classical processing error: {str(e)}")
                        return {
                            "error": str(e),
                            "response": "I encountered an error while processing your request."
                        }

                async def _classical_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
                    """Perform classical optimization."""
                    # Simple implementation for demonstration
                    start_time = datetime.now()

                    # Simulate optimization process
                    await asyncio.sleep(0.5)  # Simulate computation time

                    resource_count = params.get("resource_count", 10)
                    # Simple greedy allocation as example
                    allocation = [1] * (resource_count // 2) + [0] * (resource_count - resource_count // 2)

                    return {
                        "success": True,
                        "allocation": allocation,
                        "objective_value": sum(allocation),
                        "computation_time": (datetime.now() - start_time).total_seconds(),
                        "method_used": "classical_optimization",
                        "details": {
                            "algorithm": "Greedy optimization",
                            "iterations": resource_count * 2
                        }
                    }

                async def _classical_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
                    """Perform classical search."""
                    # Simple implementation for demonstration
                    start_time = datetime.now()

                    # Simulate search process
                    db_size = params.get("database_size", 1000)
                    await asyncio.sleep(db_size / 10000)  # Simulate computation time

                    # Simulate finding some results
                    result_count = min(5, db_size // 200)

                    return {
                        "success": True,
                        "results": [f"Result {i+1}" for i in range(result_count)],
                        "result_count": result_count,
                        "database_size": db_size,
                        "computation_time": (datetime.now() - start_time).total_seconds(),
                        "method_used": "classical_search",
                        "details": {
                            "algorithm": "Linear search",
                            "comparisons": db_size
                        }
                    }

                async def _quantum_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
                    """Perform quantum-accelerated search."""
                    # Implementation using Grover's algorithm (simulated)
                    start_time = datetime.now()

                    # Simulate quantum search with Grover's algorithm
                    db_size = params.get("database_size", 1000)

                    # Grover's algorithm provides quadratic speedup: O(√N) vs O(N)
                    classical_time = db_size / 10000
                    quantum_time = np.sqrt(db_size) / 10000

                    # Simulate the quantum computation
                    await asyncio.sleep(quantum_time)

                    # Simulate finding some results
                    result_count = min(5, db_size // 200)

                    return {
                        "success": True,
                        "results": [f"Result {i+1}" for i in range(result_count)],
                        "result_count": result_count,
                        "database_size": db_size,
                        "computation_time": (datetime.now() - start_time).total_seconds(),
                        "method_used": "quantum_grover",
                        "details": {
                            "algorithm": "Grover's search algorithm",
                            "iterations": int(np.pi/4 * np.sqrt(db_size)),
                            "quantum_advantage": f"{classical_time/quantum_time:.2f}x speedup"
                        }
                    }

                async def _generate_explanation(self, task_type: str, **kwargs) -> str:
                    """Generate a natural language explanation of the results using GPT-4o."""
                    try:
                        if task_type == "factorization":
                            number = kwargs.get("number", 0)
                            factors = kwargs.get("factors", [])
                            method = kwargs.get("method", "unknown")
                            time = kwargs.get("computation_time", 0)

                            prompt = f"""
                            The number {number} has been factored using {method} computation.
                            Factors found: {', '.join(map(str, factors))}
                            Computation time: {time:.4f} seconds

                            Please provide a clear explanation of these results, explaining the significance
                            of factorization in quantum computing and why certain methods are faster.
                            """

                        elif task_type == "optimization":
                            parameters = kwargs.get("parameters", {})
                            result = kwargs.get("result", {})
                            method = kwargs.get("method", "unknown")

                            prompt = f"""
                            An optimization problem with {len(parameters.get('variables', []))} variables 
                            has been solved using {method} computation.
                            Objective: {parameters.get('objective_type', 'Unknown')}
                            Result: {result.get('allocation', [])}
                            Objective value: {result.get('objective_value', 0)}

                            Please provide a clear explanation of these results, explaining how quantum
                            optimization can provide advantages for certain problems.
                            """

                        elif task_type == "search":
                            parameters = kwargs.get("parameters", {})
                            result = kwargs.get("result", {})
                            method = kwargs.get("method", "unknown")

                            prompt = f"""
                            A search query was processed using {method} computation.
                            Query: {parameters.get('query', 'Unknown')}
                            Database size: {parameters.get('database_size', 0)} items
                            Results found: {result.get('result_count', 0)}

                            Please provide a clear explanation of these results, explaining how quantum
                            search algorithms like Grover's can provide quadratic speedup over classical methods.
                            """

                        else:
                            return "I processed your request but don't have details about the specific method used."

                        messages = [
                            {"role": "system", "content": "You are a quantum computing expert explaining results to users."},
                            {"role": "user", "content": prompt}
                        ]

                        completion = await self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=300,
                            temperature=0.7
                        )

                        return completion.choices[0].message.content

                    except Exception as e:
                        logging.error(f"Explanation generation error: {str(e)}")
                        return f"I processed your request using {task_type} methods."

                def _update_memory(self, task: str, results: Dict[str, Any]):
                    """Update memory with task results."""
                    try:
                        self.memory.add('system', json.dumps({
                            'task': task,
                            'quantum_enabled': self.use_quantum,
                            'results': results
                        }))
                    except Exception as e:
                        logging.error(f"Error updating memory: {str(e)}")

                def get_quantum_metrics(self) -> Dict[str, Union[float, str]]:
                    """Get enhanced metrics about quantum processing."""
                    try:
                        if not self.use_quantum or not hasattr(self, 'quantum_optimizer'):
                            return {
                                'quantum_enabled': False,
                                'quantum_state': 'disabled'
                            }

                        circuit_stats = self.quantum_optimizer.get_circuit_stats()
                        return {
                            'quantum_enabled': True,
                            'processing_type': 'quantum',
                            'n_qubits': self.n_qubits,
                            'circuit_depth': circuit_stats.get('circuit_depth', 0),
                            'quantum_backend': 'Azure Quantum IonQ' if self.quantum_optimizer.use_azure else 'Quantum Simulator',
                            'max_number_size': circuit_stats.get('max_number_size', 2**(self.n_qubits//2)),
                            'theoretical_speedup': circuit_stats.get('theoretical_speedup', 'Unknown'),
                            'quantum_capabilities': [
                                'Shor\'s Algorithm (Prime Factorization)',
                                'QAOA (Optimization)',
                                'Grover\'s Algorithm (Search)',
                                'Quantum Phase Estimation'
                            ]
                        }
                    except Exception as e:
                        logging.error(f"Error getting quantum metrics: {str(e)}")
                        return {'quantum_enabled': False, 'error': str(e)}
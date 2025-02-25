"""Chain component for sequencing operations in the QUASAR framework."""

from typing import List, Callable, Any

class Chain:
    """Executes a sequence of operations in a predefined order."""
    
    def __init__(self, functions: List[Callable] = None):
        """
        Initialize the chain with a list of functions.
        
        Args:
            functions (List[Callable]): List of functions to execute in sequence
        """
        self.functions = functions or []
        
    def add(self, func: Callable) -> None:
        """Add a function to the chain."""
        self.functions.append(func)
        
    async def execute(self, input_data: Any) -> Any:
        """
        Execute the chain of functions sequentially.
        
        Args:
            input_data: Initial input to the chain
            
        Returns:
            Any: Result after executing all functions
        """
        result = input_data
        for func in self.functions:
            try:
                result = await func(result) if callable(func) else func
            except Exception as e:
                raise Exception(f"Chain execution failed at function {func.__name__}: {str(e)}")
        return result

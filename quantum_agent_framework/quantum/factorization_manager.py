"""Manages factorization methods and coordinates between classical and quantum approaches."""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import time
from openai import AsyncOpenAI

@dataclass
class FactorizationResult:
    factors: List[int]
    method_used: str
    computation_time: float
    success: bool
    details: Dict[str, Any]

class FactorizationManager:
    """Coordinates between classical and quantum factorization methods."""
    
    def __init__(self, quantum_optimizer=None):
        """Initialize the factorization manager."""
        self.quantum_optimizer = quantum_optimizer
        self.openai_client = AsyncOpenAI()
        self.classical_threshold = 1000000  # Numbers below this use classical method
        self.quantum_threshold = 2**15  # Maximum size for quantum factorization
        
    def _get_classical_factors(self, n: int) -> List[int]:
        """Get all factors of a number using classical method."""
        factors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                factors.append(i)
                if i * i != n:  # Avoid duplicating square roots
                    factors.append(n // i)
            i += 1
        return sorted(factors)
    
    async def _analyze_factorization_approach(self, number: int) -> Dict[str, Any]:
        """Use AI to analyze and recommend factorization approach."""
        try:
            messages = [
                {"role": "system", "content": f"""Analyze this number for factorization:
                Number: {number}
                Classical threshold: {self.classical_threshold}
                Quantum threshold: {self.quantum_threshold}
                
                Determine best approach based on:
                1. Number size and complexity
                2. Known factorization patterns
                3. Computational efficiency
                
                Return JSON with fields:
                - method: "classical", "quantum", or "hybrid"
                - reasoning: explanation
                - special_case: any special mathematical properties"""},
                {"role": "user", "content": f"Analyze {number} for factorization"}
            ]
            
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o",  # Latest model with optimization capabilities
                messages=messages,
                max_tokens=300,
                temperature=0.2
            )
            
            import json
            response = completion.choices[0].message.content
            return json.loads(response)
            
        except Exception as e:
            logging.error(f"AI analysis error: {str(e)}")
            # Fallback to size-based decision
            return {
                "method": "classical" if number < self.classical_threshold else "quantum",
                "reasoning": "Fallback to size-based decision",
                "special_case": None
            }
    
    async def factorize(self, number: int) -> FactorizationResult:
        """Main factorization method that chooses and executes the appropriate approach."""
        start_time = time.time()
        
        try:
            # Input validation
            if not isinstance(number, int) or number < 1:
                return FactorizationResult(
                    factors=[],
                    method_used="error",
                    computation_time=0,
                    success=False,
                    details={"error": "Invalid input: must be a positive integer"}
                )
            
            # Get AI analysis
            analysis = await self._analyze_factorization_approach(number)
            method = analysis.get("method", "classical")
            
            # Classical factorization for small numbers or when recommended
            if method == "classical" or number < self.classical_threshold:
                factors = self._get_classical_factors(number)
                return FactorizationResult(
                    factors=factors,
                    method_used="classical",
                    computation_time=time.time() - start_time,
                    success=True,
                    details={
                        "analysis": analysis,
                        "reason": "Used classical method due to number size or AI recommendation"
                    }
                )
            
            # Quantum factorization when available and recommended
            if self.quantum_optimizer and method in ("quantum", "hybrid"):
                try:
                    quantum_result = self.quantum_optimizer.factorize_number(number)
                    if quantum_result.get("success", False):
                        # Combine quantum factors with classical verification
                        quantum_factors = quantum_result.get("factors", [])
                        verified_factors = self._get_classical_factors(number)
                        
                        return FactorizationResult(
                            factors=verified_factors,  # Use complete classical factors
                            method_used="quantum",
                            computation_time=time.time() - start_time,
                            success=True,
                            details={
                                "analysis": analysis,
                                "quantum_metrics": quantum_result,
                                "quantum_factors_found": quantum_factors
                            }
                        )
                except Exception as qe:
                    logging.error(f"Quantum factorization error: {str(qe)}")
            
            # Fallback to classical for any failures or if quantum is unavailable
            factors = self._get_classical_factors(number)
            return FactorizationResult(
                factors=factors,
                method_used="classical_fallback",
                computation_time=time.time() - start_time,
                success=True,
                details={
                    "analysis": analysis,
                    "reason": "Fallback to classical due to quantum unavailability or error"
                }
            )
            
        except Exception as e:
            logging.error(f"Factorization error: {str(e)}")
            return FactorizationResult(
                factors=[],
                method_used="error",
                computation_time=time.time() - start_time,
                success=False,
                details={"error": str(e)}
            )

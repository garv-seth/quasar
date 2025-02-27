"""
API key management utilities for the QUASAR framework.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import json

# Try to import dotenv, but don't fail if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def check_api_keys() -> Dict[str, bool]:
    """
    Check which API keys are available in the environment.
    
    Returns:
        Dict with key availability status
    """
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "azure_quantum": bool(os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME")),
    }

def get_missing_keys() -> List[str]:
    """
    Get a list of missing API keys.
    
    Returns:
        List of missing key identifiers
    """
    status = check_api_keys()
    return [key for key, available in status.items() if not available]

def set_api_key(service: str, key: str) -> bool:
    """
    Set an API key in the environment.
    
    Args:
        service: Service identifier ('openai', 'anthropic', etc.)
        key: API key value
        
    Returns:
        bool: Success status
    """
    if not key:
        return False
        
    if service == "openai":
        os.environ["OPENAI_API_KEY"] = key
        return True
    elif service == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = key
        return True
    elif service == "azure_quantum":
        # For Azure Quantum, we'd need multiple settings
        # This is a simplified version
        os.environ["AZURE_QUANTUM_WORKSPACE_NAME"] = key
        return True
    else:
        logging.warning(f"Unknown service: {service}")
        return False

def save_api_keys_to_dotenv() -> bool:
    """
    Save current API keys to .env file.
    
    Returns:
        bool: Success status
    """
    try:
        # Get existing content
        env_content = ""
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                env_content = f.read()
        
        # Parse existing content into dictionary
        env_vars = {}
        for line in env_content.split("\n"):
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
        
        # Update with current environment values
        if os.environ.get("OPENAI_API_KEY"):
            env_vars["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if os.environ.get("ANTHROPIC_API_KEY"):
            env_vars["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
        if os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME"):
            env_vars["AZURE_QUANTUM_WORKSPACE_NAME"] = os.environ["AZURE_QUANTUM_WORKSPACE_NAME"]
        
        # Write back to file
        with open(".env", "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        return True
    except Exception as e:
        logging.error(f"Error saving API keys to .env: {str(e)}")
        return False

def get_api_key_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about required API keys.
    
    Returns:
        Dict with key info
    """
    return {
        "openai": {
            "name": "OpenAI API Key",
            "description": "Required for GPT-4o integration",
            "url": "https://platform.openai.com/api-keys",
            "env_var": "OPENAI_API_KEY",
            "required": False
        },
        "anthropic": {
            "name": "Anthropic API Key",
            "description": "Required for Claude 3.7 Sonnet integration",
            "url": "https://console.anthropic.com/",
            "env_var": "ANTHROPIC_API_KEY",
            "required": False
        },
        "azure_quantum": {
            "name": "Azure Quantum Workspace",
            "description": "Required for quantum hardware acceleration",
            "url": "https://azure.microsoft.com/en-us/products/quantum",
            "env_var": "AZURE_QUANTUM_WORKSPACE_NAME",
            "required": False
        }
    }
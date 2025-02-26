"""Configuration module for the QUASAR framework."""

import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv

# Load environment variables from .env file if present
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create logger
logger = logging.getLogger("quasar")


class Config:
    """Configuration manager for the QUASAR framework."""

    # Quantum settings
    QUANTUM_ENABLED = os.getenv("QUANTUM_ENABLED",
                                "true").lower() in ["true", "1", "yes"]
    DEFAULT_QUBITS = int(os.getenv("DEFAULT_QUBITS", "8"))
    MAX_QUBITS = int(os.getenv("MAX_QUBITS", "29"))  # IonQ simulator limit
    USE_AZURE = os.getenv("USE_AZURE", "true").lower() in ["true", "1", "yes"]

    # Azure Quantum settings
    AZURE_QUANTUM_SUBSCRIPTION_ID = os.getenv("AZURE_QUANTUM_SUBSCRIPTION_ID")
    AZURE_QUANTUM_RESOURCE_GROUP = os.getenv("AZURE_QUANTUM_RESOURCE_GROUP")
    AZURE_QUANTUM_WORKSPACE_NAME = os.getenv("AZURE_QUANTUM_WORKSPACE_NAME")
    AZURE_QUANTUM_LOCATION = os.getenv("AZURE_QUANTUM_LOCATION")

    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quasar.db")

    # OpenAI API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")

    # Web Agent settings
    USER_AGENT = os.getenv("USER_AGENT", "QUASAR-Agent/1.0")
    SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "30"))

    # Task processing thresholds
    FACTORIZATION_THRESHOLD = int(
        os.getenv("FACTORIZATION_THRESHOLD", "100000"))
    SEARCH_THRESHOLD = int(os.getenv("SEARCH_THRESHOLD", "1000"))
    OPTIMIZATION_THRESHOLD = int(os.getenv("OPTIMIZATION_THRESHOLD", "10"))

    # Application settings
    APP_NAME = "Q3A: Quantum-Accelerated AI Agent"
    APP_DESCRIPTION = "Quantum-Accelerated AI Agent powered by QUASAR framework"
    APP_VERSION = "1.0.0"
    DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE",
                                 "true").lower() in ["true", "1", "yes"]

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return {
            "quantum": {
                "enabled": cls.QUANTUM_ENABLED,
                "default_qubits": cls.DEFAULT_QUBITS,
                "max_qubits": cls.MAX_QUBITS,
                "use_azure": cls.USE_AZURE,
                "azure_configured": bool(cls.AZURE_QUANTUM_SUBSCRIPTION_ID)
            },
            "thresholds": {
                "factorization": cls.FACTORIZATION_THRESHOLD,
                "search": cls.SEARCH_THRESHOLD,
                "optimization": cls.OPTIMIZATION_THRESHOLD
            },
            "app": {
                "name": cls.APP_NAME,
                "version": cls.APP_VERSION,
                "development_mode": cls.DEVELOPMENT_MODE
            }
        }

    @classmethod
    def get_azure_config(cls) -> Optional[Dict[str, str]]:
        """Get Azure configuration if available."""
        if all([
                cls.AZURE_QUANTUM_SUBSCRIPTION_ID,
                cls.AZURE_QUANTUM_RESOURCE_GROUP,
                cls.AZURE_QUANTUM_WORKSPACE_NAME, cls.AZURE_QUANTUM_LOCATION
        ]):
            return {
                "subscription_id": cls.AZURE_QUANTUM_SUBSCRIPTION_ID,
                "resource_group": cls.AZURE_QUANTUM_RESOURCE_GROUP,
                "workspace_name": cls.AZURE_QUANTUM_WORKSPACE_NAME,
                "location": cls.AZURE_QUANTUM_LOCATION
            }
        return None

    @classmethod
    def save_to_file(cls, filename: str = "config.json") -> bool:
        """Save configuration to a file."""
        try:
            # Create a sanitized config (no sensitive data)
            config_data = cls.to_dict()

            with open(filename, "w") as f:
                json.dump(config_data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error saving config to file: {str(e)}")
            return False

    @classmethod
    def is_quantum_available(cls) -> bool:
        """Check if quantum computing is available."""
        return cls.QUANTUM_ENABLED and (cls.USE_AZURE
                                        and cls.get_azure_config() is not None
                                        or not cls.USE_AZURE)

    @classmethod
    def get_openai_config(cls) -> Dict[str, str]:
        """Get OpenAI configuration."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "default_model": cls.DEFAULT_MODEL
        }

    @classmethod
    def validate(cls) -> Dict[str, bool]:
        """Validate configuration and return status."""
        return {
            "quantum_enabled": cls.QUANTUM_ENABLED,
            "azure_configured": cls.get_azure_config() is not None,
            "database_configured": bool(cls.DATABASE_URL),
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "development_mode": cls.DEVELOPMENT_MODE
        }


def create_sample_env_file() -> bool:
    """Create a sample .env file if not exists."""
    if os.path.exists(".env"):
        return False

    sample_env = """# QUASAR Configuration

# Quantum Settings
QUANTUM_ENABLED=true
DEFAULT_QUBITS=8
MAX_QUBITS=29
USE_AZURE=true

# Azure Quantum Settings
AZURE_QUANTUM_SUBSCRIPTION_ID=your-subscription-id
AZURE_QUANTUM_RESOURCE_GROUP=your-resource-group
AZURE_QUANTUM_WORKSPACE_NAME=your-workspace-name
AZURE_QUANTUM_LOCATION=your-location

# Database Settings
DATABASE_URL=sqlite:///./quasar.db

# OpenAI API Settings
OPENAI_API_KEY=your-openai-api-key
DEFAULT_MODEL=gpt-4o

# Web Agent Settings
USER_AGENT=QUASAR-Agent/1.0
SEARCH_TIMEOUT=30

# Task Processing Thresholds
FACTORIZATION_THRESHOLD=100000
SEARCH_THRESHOLD=1000
OPTIMIZATION_THRESHOLD=10

# Application Settings
DEVELOPMENT_MODE=true
LOG_LEVEL=INFO
"""

    try:
        with open(".env", "w") as f:
            f.write(sample_env)
        logger.info("Created sample .env file")
        return True
    except Exception as e:
        logger.error(f"Error creating sample .env file: {str(e)}")
        return False


# Create sample .env file if not exists
if not os.path.exists(".env"):
    create_sample_env_file()

# Check configuration validity
config_status = Config.validate()
if not all(config_status.values()):
    logger.warning("Some configuration settings are missing or invalid:")
    for key, value in config_status.items():
        if not value:
            logger.warning(f"- {key} is not properly configured")
else:
    logger.info("Configuration validated successfully")

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

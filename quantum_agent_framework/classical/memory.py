from typing import List, Dict, Any
from datetime import datetime

class Memory:
    """Manages conversation history and context."""
    
    def __init__(self, max_messages: int = 100):
        """
        Initialize memory storage.
        
        Args:
            max_messages (int): Maximum number of messages to store
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        
    def add(self, role: str, content: str) -> None:
        """
        Add a message to memory.
        
        Args:
            role (str): Role of the message sender (e.g., 'user', 'assistant')
            content (str): Message content
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
            
    def get_history(self) -> List[Dict[str, Any]]:
        """Get all stored messages."""
        return self.messages
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = []

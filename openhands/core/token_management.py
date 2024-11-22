from typing import List, Optional
from openhands.events.event import Event
from openhands.core.logger import openhands_logger as logger

class TokenManager:
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, llm):
        self.llm = llm
        self.max_input_tokens = self._get_max_input_tokens()

    def _get_max_input_tokens(self) -> int:
        # Priority: Model info > Config > Environment > Default
        if hasattr(self.llm, 'model_info') and self.llm.model_info.get('max_input_tokens'):
            return self.llm.model_info['max_input_tokens']
        
        if self.llm.config.max_input_tokens is not None:
            return self.llm.config.max_input_tokens

        # Check environment variable
        import os
        env_tokens = os.getenv('OPENHANDS_MAX_INPUT_TOKENS')
        if env_tokens and env_tokens.isdigit():
            return int(env_tokens)

        return self.DEFAULT_MAX_TOKENS

    def check_token_limit(self, events: List[Event], memory: Optional[str] = None) -> bool:
        """Check if adding these events would exceed token limit.
        
        Args:
            events: List of events to check
            memory: Optional memory string to include in token count
            
        Returns:
            bool: True if within limit, False if would exceed
        """
        try:
            total_tokens = self.llm.get_token_count(events)
            if memory:
                total_tokens += self.llm.get_token_count(memory)
                
            return total_tokens <= self.max_input_tokens
            
        except Exception as e:
            logger.error(f'Error checking token count: {e}')
            return True  # Allow if we can't check

    def get_token_count(self, events: List[Event], memory: Optional[str] = None) -> int:
        """Get total token count for events and memory.
        
        Args:
            events: List of events to count
            memory: Optional memory string to include
            
        Returns:
            int: Total token count
        """
        try:
            total = self.llm.get_token_count(events)
            if memory:
                total += self.llm.get_token_count(memory)
            return total
        except Exception as e:
            logger.error(f'Error getting token count: {e}')
            return 0
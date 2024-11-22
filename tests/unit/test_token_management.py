import os
import pytest
from unittest.mock import MagicMock, patch

from openhands.core.token_management import TokenManager
from openhands.events.action import MessageAction
from openhands.events.event import Event, EventSource

def test_token_manager_initialization():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = 1000
    token_manager = TokenManager(mock_llm)
    assert token_manager.max_input_tokens == 1000

def test_token_manager_env_var_override():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = None
    with patch.dict(os.environ, {'OPENHANDS_MAX_INPUT_TOKENS': '2000'}):
        token_manager = TokenManager(mock_llm)
        assert token_manager.max_input_tokens == 2000

def test_token_manager_default_fallback():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = None
    token_manager = TokenManager(mock_llm)
    assert token_manager.max_input_tokens == TokenManager.DEFAULT_MAX_TOKENS

def test_token_manager_check_limit():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = 100
    mock_llm.get_token_count.return_value = 50
    
    token_manager = TokenManager(mock_llm)
    
    events = [
        MessageAction(content="Test message", wait_for_response=False)
    ]
    events[0]._source = EventSource.USER
    
    assert token_manager.check_token_limit(events)
    mock_llm.get_token_count.assert_called_once()

def test_token_manager_check_limit_with_memory():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = 100
    mock_llm.get_token_count.side_effect = [50, 30]  # Events count, memory count
    
    token_manager = TokenManager(mock_llm)
    
    events = [
        MessageAction(content="Test message", wait_for_response=False)
    ]
    events[0]._source = EventSource.USER
    memory = "Previous context"
    
    assert token_manager.check_token_limit(events, memory)
    assert mock_llm.get_token_count.call_count == 2

def test_token_manager_exceeds_limit():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = 100
    mock_llm.get_token_count.return_value = 150
    
    token_manager = TokenManager(mock_llm)
    
    events = [
        MessageAction(content="Test message", wait_for_response=False)
    ]
    events[0]._source = EventSource.USER
    
    assert not token_manager.check_token_limit(events)

def test_token_manager_error_handling():
    mock_llm = MagicMock()
    mock_llm.config.max_input_tokens = 100
    mock_llm.get_token_count.side_effect = Exception("Token counting failed")
    
    token_manager = TokenManager(mock_llm)
    
    events = [
        MessageAction(content="Test message", wait_for_response=False)
    ]
    events[0]._source = EventSource.USER
    
    # Should return True when token counting fails
    assert token_manager.check_token_limit(events)
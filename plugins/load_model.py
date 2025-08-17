import logging

from types import SimpleNamespace
from typing import TypedDict
from sublime import View, load_settings

logger = logging.getLogger(__name__)


class AssistantSettings(TypedDict):
    name: str
    chat_model: str
    assistant_role: str
    temperature: float
    token: str
    url: str
    timeout: int


def get_model_or_default(view: View) -> AssistantSettings:
    """Get LLM settings for code completion."""
    settings = load_settings('llm-completion.sublime-settings')
    
    try:
        llm_settings = settings.get('llm_settings', {})
        if not llm_settings:
            raise RuntimeError("No llm_settings configured in settings")
        
        # Convert the simplified settings to AssistantSettings format
        assistant_config = {
            'name': 'LLM Completion',
            'chat_model': llm_settings.get('model', 'gpt-4o-mini'),
            'assistant_role': llm_settings.get('system_message', 'You are a senior developer who writes clean, correct code. Only return the completion text without explanation.'),
            'temperature': llm_settings.get('temperature', 0.2),
            'token': llm_settings.get('token', 'YOUR_API_TOKEN_HERE'),
            'url': llm_settings.get('url', 'https://api.openai.com/v1/chat/completions'),
            'timeout': llm_settings.get('timeout', 30)
        }
        
        assistant = SimpleNamespace(**assistant_config)
        logger.debug('Using LLM: %s', assistant_config['chat_model'])
        return assistant
        
    except (RuntimeError, KeyError) as error:
        logger.error('Error loading LLM settings: %s', error)
        # Return a minimal default
        default_config = {
            'name': 'Default LLM',
            'chat_model': 'gpt-4o-mini',
            'assistant_role': 'You are a senior developer who writes clean, correct code. Only return the completion text without explanation.',
            'temperature': 0.2,
            'token': 'YOUR_API_TOKEN_HERE',
            'url': 'https://api.openai.com/v1/chat/completions',
            'timeout': 30
        }
        return SimpleNamespace(**default_config)
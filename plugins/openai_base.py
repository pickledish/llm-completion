from __future__ import annotations

import json
import logging
import threading
import urllib.request
from typing import Callable

import sublime
from .load_model import AssistantSettings

logger = logging.getLogger(__name__)


class CommonMethods:
    _completion_thread = None

    @classmethod
    def request_inline_completion(
        cls, 
        view, 
        assistant: AssistantSettings, 
        prompt: str, 
        callback: Callable[[str], None]
    ):
        """Request inline completion from LLM using direct API call."""
        print(f"[LLM_COMPLETION] Starting completion request")
        
        # Cancel any existing completion request
        cls.stop_completion_worker()
        
        def make_api_call():
            try:
                # Prepare the request data
                request_data = {
                    "model": assistant.chat_model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": assistant.assistant_role or "You are a senior developer who writes clean, correct code. Only return the completion text without explanation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": getattr(assistant, 'temperature', 0.2),
                    "stream": False
                }

                # Prepare headers
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {assistant.token}'
                }
                
                # Create the request
                url = assistant.url or "https://api.openai.com/v1/chat/completions"
                
                request = urllib.request.Request(
                    url,
                    data=json.dumps(request_data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                
                # Make the request
                timeout = getattr(assistant, 'timeout', 30)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                    
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        completion_text = response_data['choices'][0]['message']['content']
                        print(f"[LLM_COMPLETION] API call successful, length: {len(completion_text)}")
                        
                        # Call the callback with the result
                        sublime.set_timeout(lambda: callback(completion_text), 0)
                    else:
                        print(f"[LLM_COMPLETION] API call failed - no choices in response")
                        
            except Exception as e:
                print(f"[LLM_COMPLETION] API call failed: {e}")
                logger.error(f"LLM completion API call failed: {e}")
        
        # Start the API call in a background thread
        cls._completion_thread = threading.Thread(target=make_api_call, daemon=True)
        cls._completion_thread.start()

    @classmethod
    def stop_completion_worker(cls):
        """Stop any running completion thread."""
        # Note: We can't cleanly cancel urllib requests, but daemon threads will exit with the main process
        cls._completion_thread = None
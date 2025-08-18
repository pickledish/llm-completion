from __future__ import annotations

import html
import logging
import re
import threading
import time
from typing import Any, Dict, Optional

import sublime
import sublime_plugin
from sublime import Region, View

from .load_model import get_model_or_default
from .openai_base import CommonMethods

logger = logging.getLogger(__name__)

COMPLETION_PROMPT_TEMPLATE = """I will give you a code snippet with a part missing, represented by the character █.

Please rewrite the code exactly as I showed you, except with the missing part represented by █ should be filled in.

Focus on a solution which is concise, straightforward, simple, and above all, correct syntax.

Be sure to rewrite the entire snippet.

Your response should never contain a █ symbol.

Here is an example:

## INPUT

```
  const newConnection = async ({
    hostname,
    port,
    username,
    password,
  } : {
    hostname: string;
    port: number;
    █
    password: string;
  }) => {
    let endpoint = `${hostname}:${port}`;
```

## You respond:

  const newConnection = async ({
    hostname,
    port,
    username,
    password,
  } : {
    hostname: string;
    port: number;
    username: string;
    password: string;
  }) => {
    let endpoint = `${hostname}:${port}`;

Okay, now here is the real task.

## INPUT

```
{context_with_placeholder}
```
"""


class InlineCompletionManager:
    def __init__(self, view: View):
        self.view = view
        self._phantom_set: Optional[sublime.PhantomSet] = None
        self._current_completion: Optional[str] = None
        self._completion_point: Optional[int] = None
        self._request_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        
    def _get_phantom_set(self) -> sublime.PhantomSet:
        if self._phantom_set is None:
            self._phantom_set = sublime.PhantomSet(self.view, "llm_inline_completion")
        return self._phantom_set
        
    def get_context(self, cursor_point: int, lines_before: Optional[int] = None, lines_after: Optional[int] = None) -> str:
        """Get context around cursor position for completion request."""
        # Get settings for context lines
        settings = sublime.load_settings('llm-completion.sublime-settings')
        if lines_before is None:
            lines_before = settings.get('inline_completion_context_before', 10)
        if lines_after is None:
            lines_after = settings.get('inline_completion_context_after', 5)
            
        current_line_region = self.view.line(cursor_point)
        current_line = self.view.rowcol(cursor_point)[0]
        
        # Get lines before
        start_line = max(0, current_line - lines_before)
        start_point = self.view.text_point(start_line, 0)
        
        # Get lines after  
        end_line = min(self.view.rowcol(self.view.size())[0], current_line + lines_after)
        end_point = self.view.text_point(end_line + 1, 0) if end_line < self.view.rowcol(self.view.size())[0] else self.view.size()
        
        # Get text before cursor
        before_cursor = self.view.substr(Region(start_point, cursor_point))
        
        # Get text after cursor  
        after_cursor = self.view.substr(Region(cursor_point, end_point))
        
        # Return context with cursor position marked
        return f"{before_cursor}<CURSOR>{after_cursor}"
        
    def should_trigger_completion(self, cursor_point: int) -> bool:
        """Check if we should trigger completion at current cursor position."""
        print(f"[LLM_COMPLETION] Checking if should trigger completion at point: {cursor_point}")
            
        # Don't trigger if we just completed something at this position
        if self._completion_point == cursor_point and self._current_completion:
            print(f"[LLM_COMPLETION] Already completed at this position, skipping")
            return False
            
        # Check character before cursor - don't trigger after logical line endings
        if cursor_point > 0:
            char_before = self.view.substr(cursor_point - 1)
            logical_endings = {',', '}', ')', ']', ';', ':', '.'}
            if char_before in logical_endings:
                print(f"[LLM_COMPLETION] Character before cursor is logical ending: '{char_before}', skipping")
                return False
        
        print(f"[LLM_COMPLETION] Should trigger completion: True")
        return True
        
    def request_completion(self, cursor_point: int):
        """Request completion from OpenAI with debouncing."""
        print(f"[LLM_COMPLETION] request_completion called for cursor_point: {cursor_point}")
        
        self.cancel_completion()
        
        if not self.should_trigger_completion(cursor_point):
            print(f"[LLM_COMPLETION] should_trigger_completion returned False, aborting")
            return
            
        print(f"[LLM_COMPLETION] Starting new completion request thread")
        self._cancel_event = threading.Event()
        self._request_thread = threading.Thread(
            target=self._request_completion_async,
            args=(cursor_point,),
            daemon=True
        )
        self._request_thread.start()
        
    def _request_completion_async(self, cursor_point: int):
        """Async completion request with debouncing."""
        print(f"[LLM_COMPLETION] _request_completion_async started for cursor_point: {cursor_point}")
        try:
            # Wait for debounce period (500ms)
            if self._cancel_event.wait(0.5):
                print(f"[LLM_COMPLETION] Request was cancelled during debounce")
                return
                
            # Double-check that cursor is still at same position
            if len(self.view.sel()) != 1 or self.view.sel()[0].begin() != cursor_point:
                print(f"[LLM_COMPLETION] Cursor position changed, aborting")
                return
            
            # Get context for completion
            context = self.get_context(cursor_point)
            print(f"[LLM_COMPLETION] Context retrieved, length: {len(context)}")
            
            # Get assistant settings
            assistant = get_model_or_default(self.view)
            
            if not assistant:
                print(f"[LLM_COMPLETION] No assistant found, aborting")
                return
            
            # Replace <CURSOR> with █ for the new prompt format
            context_with_placeholder = context.replace('<CURSOR>', '█')
            
            # Prepare completion prompt using the template
            completion_prompt = COMPLETION_PROMPT_TEMPLATE.format(context_with_placeholder=context_with_placeholder)
            
            print(f"[LLM_COMPLETION] Making OpenAI request")
            
            # Make request through CommonMethods
            sublime.set_timeout_async(
                lambda: self._make_openai_request(assistant, completion_prompt, cursor_point, context_with_placeholder),
                0
            )
            
        except Exception as e:
            print(f"[LLM_COMPLETION] Error in completion request: {e}")
            logger.error(f"Error in completion request: {e}")
            
    def _make_openai_request(self, assistant, prompt: str, cursor_point: int, context_with_placeholder: str):
        """Make the actual OpenAI API request."""
        if self._cancel_event.is_set():
            return
            
        # Create a completion handler that extracts just the filled-in part
        context_for_extraction = context_with_placeholder
        
        def completion_handler(response_text: str):
            print(f"[LLM_COMPLETION] Completion handler called with response length: {len(response_text)}")
            
            if not self._cancel_event.is_set():
                # Extract the completion part from the response
                extracted_completion = self._extract_completion_from_response(
                    response_text, 
                    context_for_extraction, 
                    cursor_point
                )
                
                if extracted_completion:
                    print(f"[LLM_COMPLETION] Extracted completion: '{extracted_completion[:50]}...'")
                    sublime.set_timeout(
                        lambda: self._show_completion(extracted_completion, cursor_point),
                        0
                    )
                
        # Make request through CommonMethods
        try:
            CommonMethods.request_inline_completion(
                self.view, 
                assistant, 
                prompt, 
                completion_handler
            )
        except Exception as e:
            print(f"[LLM_COMPLETION] Error making OpenAI completion request: {e}")
            logger.error(f"Error making OpenAI completion request: {e}")
        
    def _show_completion(self, completion_text: str, cursor_point: int):
        """Display completion as phantom."""
        if not completion_text or self._cancel_event.is_set():
            return
            
        # Verify cursor is still at expected position
        if len(self.view.sel()) != 1 or self.view.sel()[0].begin() != cursor_point:
            return
        
        # Clean up completion text but preserve necessary whitespace
        cleaned_completion = self._clean_completion_text(completion_text)
        
        if not cleaned_completion:
            return
        
        self._current_completion = cleaned_completion
        self._completion_point = cursor_point
        
        # Create phantom content
        phantom_html = self._create_phantom_html(cleaned_completion)
        
        # Follow LSP-copilot approach: two phantoms for multi-line completions
        lines = cleaned_completion.splitlines()
        first_line = lines[0] if lines else ""
        rest_lines = lines[1:] if len(lines) > 1 else []
        
        phantoms = []
        
        # First phantom: first line only, positioned at cursor + 1 with LAYOUT_INLINE
        if first_line:
            first_phantom_html = self._create_phantom_html(first_line)
            phantoms.append(sublime.Phantom(
                Region(cursor_point, cursor_point),
                first_phantom_html,
                sublime.LAYOUT_INLINE
            ))
        
        # Second phantom: rest of lines at cursor point with LAYOUT_BLOCK
        # (Required even if empty to prevent cursor jumping)
        rest_phantom_html = self._create_phantom_html('\n'.join(rest_lines) if rest_lines else "")
        phantoms.append(sublime.Phantom(
            Region(cursor_point, cursor_point),
            rest_phantom_html,
            sublime.LAYOUT_BLOCK
        ))
        
        phantom_set = self._get_phantom_set()
        phantom_set.update(phantoms)
    
    def _clean_completion_text(self, completion_text: str) -> str:
        """Clean completion text while preserving important whitespace."""
        # Remove markdown code blocks if present
        cleaned = completion_text
        if cleaned.strip().startswith('```'):
            lines = cleaned.split('\n')
            if len(lines) > 2 and lines[-1].strip() == '```':
                # Remove first and last lines (```language and ```)
                cleaned = '\n'.join(lines[1:-1])
            else:
                # Malformed code block, just remove the first ```
                cleaned = cleaned.strip()
                if cleaned.startswith('```'):
                    first_newline = cleaned.find('\n')
                    if first_newline != -1:
                        cleaned = cleaned[first_newline + 1:]
                
        # Only trim trailing whitespace, preserve leading whitespace and internal structure
        cleaned = cleaned.rstrip()
        
        return cleaned
    
    def _extract_completion_from_response(self, response_text: str, original_context: str, cursor_point: int) -> str:
        """Extract just the completion part from the LLM response."""
        
        # First, clean the response to remove markdown code blocks
        cleaned_response = self._clean_completion_text(response_text)
        
        # Find the █ character position in the original context
        placeholder_pos = original_context.find('█')
        if placeholder_pos == -1:
            return ""
        
        # Split the original context into before and after parts
        context_before = original_context[:placeholder_pos]
        context_after = original_context[placeholder_pos + 1:]
        
        # Strip whitespace from both context and response for better matching
        stripped_context_before = context_before.strip()
        stripped_cleaned_response = cleaned_response.strip()

        # Use regex to find where the context before ends and completion should start
        escaped_context_before = re.escape(stripped_context_before)

        # Try full context first
        match = re.search(escaped_context_before, stripped_cleaned_response)
        if match:
            completion_start = match.end()
        else:
            # Try fallback: last 30 characters of context_before
            fallback_context = stripped_context_before[-30:] if len(stripped_context_before) > 30 else stripped_context_before
            escaped_fallback = re.escape(fallback_context)
            
            match = re.search(escaped_fallback, stripped_cleaned_response)
            if match:
                completion_start = match.end()
            else:
                # Final fallback: assume response starts with completion
                completion = stripped_cleaned_response
                if len(completion) > 500:
                    return "█"
                return completion
        
        # Look for the after part to determine where completion ends
        after_pos = -1
        if context_after.strip():  # Only look for after context if it's not empty/whitespace
            stripped_context_after = context_after.strip()
            escaped_context_after = re.escape(stripped_context_after)

            # Try full context_after first
            after_match = re.search(escaped_context_after, stripped_cleaned_response[completion_start:])
            if after_match:
                after_pos = completion_start + after_match.start()
            else:
                # Try fallback: first 30 characters of context_after
                fallback_after_context = stripped_context_after[:30] if len(stripped_context_after) > 30 else stripped_context_after
                escaped_fallback_after = re.escape(fallback_after_context)
                
                after_match = re.search(escaped_fallback_after, stripped_cleaned_response[completion_start:])
                if after_match:
                    after_pos = completion_start + after_match.start()
        
        # Extract the completion
        if after_pos != -1:
            # Found both before and after context - extract between them
            completion = stripped_cleaned_response[completion_start:after_pos]
        else:
            # No after context found - this is common when LLM provides good completion but doesn't reproduce after_context
            completion = stripped_cleaned_response[completion_start:]
            
            # Since we successfully matched before_context, be more lenient with completion length
            # Only reject if it's extremely long (likely indicating the LLM reproduced lots of unrelated content)
            if len(completion.strip()) > 2000:  # More permissive threshold since we have good before_context anchor
                return "█"

        # Clean up the completion
        completion = completion.strip()
        
        # Validate that we actually extracted something meaningful
        if not completion:
            return "█"

        # Check if the completion is suspiciously long (might indicate extraction failed)
        if len(completion) > 1000:  # Arbitrary threshold for "too long"
            return "█"

        return completion
        
    def _create_phantom_html(self, completion_text: str) -> str:
        """Create HTML for completion phantom."""
        # Escape HTML and handle whitespace carefully 
        escaped_text = html.escape(completion_text)
        tab_size = self.view.settings().get("tab_size", 4)
        
        # Replace tabs with spaces, but use &nbsp; only for leading spaces on each line
        escaped_text = escaped_text.replace("\t", " " * tab_size)
        
        # Convert leading spaces to &nbsp; to preserve indentation in HTML
        lines = escaped_text.split('\n')
        processed_lines = []
        for line in lines:
            # Find leading spaces
            leading_spaces = len(line) - len(line.lstrip(' '))
            if leading_spaces > 0:
                # Replace leading spaces with &nbsp;, keep the rest as regular spaces
                line = '&nbsp;' * leading_spaces + line[leading_spaces:]
            processed_lines.append(line)
        escaped_text = '\n'.join(processed_lines)
        
        # Split into lines for proper display
        lines = escaped_text.split('\n')
        
        # Simple approach - just display all content as pre-formatted text
        body = f'<span class="completion-text">{escaped_text}</span>'
        
        # Create HTML template with pre-formatted text
        html_template = f"""<body id="llm-inline-completion">
<style>
body {{
    color: #808080;
    font-style: italic;
    margin: 0;
    padding: 0;
}}
.completion-text {{
    color: #808080;
    white-space: pre;
    display: inline;
}}
</style>
{body}
</body>"""
        
        return html_template
        
    def accept_completion(self, edit=None) -> bool:
        """Accept the current completion."""
        print(f"[LLM_COMPLETION] accept_completion called")
        
        if not self._current_completion or not self._completion_point:
            print(f"[LLM_COMPLETION] No completion to accept")
            return False
        
        print(f"[LLM_COMPLETION] Accepting completion: '{self._current_completion[:50]}...'")
        
        # Always use edit object for raw insertion to avoid auto-indentation issues
        if edit is None:
            print(f"[LLM_COMPLETION] No edit object provided - this should not happen with TextCommand")
            return False
        else:
            self.view.insert(edit, self._completion_point, self._current_completion)
            
        self.hide_completion()
        return True
        
    def hide_completion(self):
        """Hide current completion."""
        if self._phantom_set:
            self._phantom_set.update([])
        self._current_completion = None
        self._completion_point = None
        
    def cancel_completion(self):
        """Cancel any pending completion request."""
        self._cancel_event.set()
        if self._request_thread and self._request_thread.is_alive():
            self._request_thread.join(timeout=0.1)
        # Also cancel any API request
        CommonMethods.stop_completion_worker()
        self.hide_completion()
        
    @property 
    def has_completion(self) -> bool:
        """Check if there's a visible completion."""
        return self._current_completion is not None


# Global manager instances for each view
_view_managers: Dict[int, InlineCompletionManager] = {}


def get_completion_manager(view: View) -> InlineCompletionManager:
    """Get or create completion manager for view."""
    view_id = view.id()
    if view_id not in _view_managers:
        _view_managers[view_id] = InlineCompletionManager(view)
    return _view_managers[view_id]


class InlineCompletionListener(sublime_plugin.ViewEventListener):
    """Event listener for inline completions."""
    
    def __init__(self, view: View):
        super().__init__(view)
        self._last_change_time = 0
        self._settings = sublime.load_settings('llm-completion.sublime-settings')
        self._last_command = None
        
    @classmethod
    def applies_to_primary_view_only(cls) -> bool:
        return False
        
    def _is_completion_enabled(self) -> bool:
        """Check if inline completion is enabled."""
        return self._settings.get('inline_completion_enabled', True)
        
    def _get_completion_delay(self) -> float:
        """Get completion delay in seconds."""
        return self._settings.get('inline_completion_delay', 0.5)
        
    def on_modified_async(self):
        """Handle text modifications."""
        if not self._is_completion_enabled():
            return
        
        # Don't trigger if the last command was accepting a completion
        if self._last_command == "accept_inline_completion":
            self._last_command = None  # Reset after use
            return
            
        manager = get_completion_manager(self.view)
        
        # Hide any existing completion on modification
        manager.hide_completion()
        
        # Only trigger on single cursor
        if len(self.view.sel()) != 1:
            return
            
        cursor_point = self.view.sel()[0].begin()
        
        # Request completion after delay
        self._last_change_time = time.time()
        delay = int(self._get_completion_delay() * 1000)
        
        sublime.set_timeout_async(
            lambda: self._delayed_completion_request(cursor_point, self._last_change_time),
            delay
        )
        
    def _delayed_completion_request(self, cursor_point: int, request_time: float):
        """Make completion request if no more recent changes."""
        if request_time != self._last_change_time:
            return  # More recent change occurred
            
        manager = get_completion_manager(self.view)
        manager.request_completion(cursor_point)
        
    def on_selection_modified_async(self):
        """Handle cursor movement."""
        manager = get_completion_manager(self.view)
        manager.hide_completion()
        
    def on_deactivated_async(self):
        """Handle view deactivation."""
        manager = get_completion_manager(self.view)
        manager.hide_completion()
        
    def on_close(self):
        """Handle view close."""
        view_id = self.view.id()
        if view_id in _view_managers:
            _view_managers[view_id].cancel_completion()
            del _view_managers[view_id]
            
    def on_query_context(self, key: str, operator: int, operand: Any, match_all: bool) -> Optional[bool]:
        """Handle context queries for key bindings."""
        if key == "llm_inline_completion_visible":
            manager = get_completion_manager(self.view)
            has_completion = manager.has_completion
            
            if operator == sublime.OP_EQUAL:
                return has_completion == operand
            elif operator == sublime.OP_NOT_EQUAL:
                return has_completion != operand
                
        return None
    
    def on_text_command(self, command_name: str, args: dict) -> None:
        """Track text commands to detect completion acceptance."""
        self._last_command = command_name


class AcceptInlineCompletionCommand(sublime_plugin.TextCommand):
    """Command to accept inline completion."""
    
    def run(self, edit):
        manager = get_completion_manager(self.view)
        manager.accept_completion(edit)
        
    def is_enabled(self):
        manager = get_completion_manager(self.view)
        return manager.has_completion


class DismissInlineCompletionCommand(sublime_plugin.TextCommand):
    """Command to dismiss inline completion."""
    
    def run(self, edit):
        manager = get_completion_manager(self.view)
        manager.hide_completion()
        
    def is_enabled(self):
        manager = get_completion_manager(self.view)
        return manager.has_completion


class ToggleLlmCompletionCommand(sublime_plugin.ApplicationCommand):
    """Command to toggle LLM completion on/off."""
    
    def run(self):
        settings = sublime.load_settings('llm-completion.sublime-settings')
        current_state = settings.get('inline_completion_enabled', True)
        new_state = not current_state
        settings.set('inline_completion_enabled', new_state)
        sublime.save_settings('llm-completion.sublime-settings')
        
        # Hide any existing completions when disabling
        if not new_state:
            for manager in _view_managers.values():
                manager.hide_completion()
        
        # Show status message
        status = "enabled" if new_state else "disabled"
        sublime.status_message(f"LLM Completion {status}")
    
    def description(self):
        settings = sublime.load_settings('llm-completion.sublime-settings')
        enabled = settings.get('inline_completion_enabled', True)
        return "Disable LLM Completion" if enabled else "Enable LLM Completion"


class EnableLlmCompletionCommand(sublime_plugin.ApplicationCommand):
    """Command to enable LLM completion."""
    
    def run(self):
        settings = sublime.load_settings('llm-completion.sublime-settings')
        settings.set('inline_completion_enabled', True)
        sublime.save_settings('llm-completion.sublime-settings')
        sublime.status_message("LLM Completion enabled")
    
    def is_enabled(self):
        settings = sublime.load_settings('llm-completion.sublime-settings')
        return not settings.get('inline_completion_enabled', True)


class DisableLlmCompletionCommand(sublime_plugin.ApplicationCommand):
    """Command to disable LLM completion."""
    
    def run(self):
        settings = sublime.load_settings('llm-completion.sublime-settings')
        settings.set('inline_completion_enabled', False)
        sublime.save_settings('llm-completion.sublime-settings')
        
        # Hide any existing completions
        for manager in _view_managers.values():
            manager.hide_completion()
            
        sublime.status_message("LLM Completion disabled")
    
    def is_enabled(self):
        settings = sublime.load_settings('llm-completion.sublime-settings')
        return settings.get('inline_completion_enabled', True)

import sys

# clear modules cache if package is reloaded (after update?)
prefix = __package__ + '.plugins'  # type: ignore # don't clear the base package
for module_name in [module_name for module_name in sys.modules if module_name.startswith(prefix)]:
    del sys.modules[module_name]
del prefix

# LLM Completion Plugin - Core imports
from .plugins.inline_completion import (  # noqa: E402, F401
    InlineCompletionListener,
    AcceptInlineCompletionCommand,
    DismissInlineCompletionCommand,
    ToggleLlmCompletionCommand,
    EnableLlmCompletionCommand,
    DisableLlmCompletionCommand,
)

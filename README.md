# LLM Completion for Sublime Text

A lightweight LLM-powered inline code completion plugin for Sublime Text that provides "phantom" / tab completions using the OpenAI (or any OpenAI-compatible) API.

## Disclaimer

This was my first (and probably last, lol) attempt at "vibe coding" -- which is to say, every line of code in this repo was written by Claude Code, I have only read bits and pieces of it. Even most of this README (not this part) was written by Claude (though I did do a correctness pass on the README, it's all accurate).

In particular, the code is based mostly on these two other plugins, whose code I gave as examples to CC:

* [OpenAI-sublime-text](https://github.com/yaroslavyaroslav/OpenAI-sublime-text)
* [LSP-Copilot](https://github.com/TerminalFi/LSP-copilot)

And then I told it "the first one talks to OpenAI, the second one does phantom completions, please make a plugin that does both", and iterated a few times.

So, big thank you to those two projects for doing all the actual work here!

## Features

- **Inline Code Completion**: Get AI-powered code suggestions as you type
- **Intelligent Context**: Uses surrounding code context for better completions
- **Multiple AI Providers**: Works with OpenAI, local LLMs via e.g. Ollama, and other OpenAI-compatible APIs
- **Configurable**: Adjustable completion delay, context size, and AI model settings
- **Lightweight**: Focused solely on code completion without bloat

## Installation

Requires Sublime Text build `4050` or newer (python 3.8).

**Manual Installation** -- I haven't uploaded this plugin to Package Control, so this is the only way:

1. Go to `Preferences` -> `Browse Packages`
2. Clone this repository into that directory:
   ```bash
   git clone https://github.com/pickledish/llm-completion.git "LLM Completion"
   ```
3. Restart Sublime Text

## Configuration (Remote)

1. Open `Preferences` -> `Package Settings` -> `LLM Completion` -> `Settings`
2. Configure your AI provider:

```json
{
    "inline_completion_enabled": true,
    "inline_completion_delay": 0.5,
    "llm_settings": {
        "model": "gpt-4o-mini",
        "token": "your-openai-api-key-here",
        "url": "https://api.openai.com/v1/chat/completions",
        "temperature": 0.2
    }
}
```

## Configuration (Local)

For using local models via Ollama (LM Studio etc also work):

```json
{
    "inline_completion_enabled": true,
    "llm_settings": {
        "model": "codellama:7b",
        "url": "http://localhost:11434/v1/chat/completions",
        "temperature": 0.2
    }
}
```

## Usage

1. Start typing code in any file
2. Stop typing for the configured period (see above)
3. You'll see AI-generated completions appear as grayed-out text
4. Press `Tab` to accept the completion
5. Press `Escape` to dismiss it

## Troubleshooting

1. Check that `inline_completion_enabled` is `true` in settings
2. Verify your API key and URL are correct
3. Check the Sublime Text console (`View` -> `Show Console`) for errors
4. Ensure you're not typing after logical line endings

## Contributing

Honestly, you're best off forking this repo and then doing whatever you want!

This was made to scratch a specific itch I had, and now that it does that, I probably won't work on it any further -- I just wanted to publish the result here in case it's useful to anyone else.

Besides, everyone knows it's no fun to maintain somebody else's code -- and I didn't write this!

But, if you leave an issue on this repository, I will make an effort to get back to you eventually.

#!/usr/bin/env python3

import re
import html

def clean_completion_text(completion_text: str) -> str:
    """Clean completion text while preserving important whitespace."""
    cleaned = completion_text
    if cleaned.strip().startswith('```'):
        lines = cleaned.split('\n')
        if len(lines) > 2 and lines[-1].strip() == '```':
            cleaned = '\n'.join(lines[1:-1])
        else:
            cleaned = cleaned.strip()
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
    cleaned = cleaned.rstrip()
    return cleaned

def extract_completion_from_response(response_text: str, original_context: str, cursor_point: int) -> str:
    """Extract just the completion part from the LLM response."""
    
    cleaned_response = clean_completion_text(response_text)
    
    placeholder_pos = original_context.find('█')
    if placeholder_pos == -1:
        return ""
    
    context_before = original_context[:placeholder_pos]
    context_after = original_context[placeholder_pos + 1:]
    
    stripped_context_before = context_before.strip()
    stripped_cleaned_response = cleaned_response.strip()

    escaped_context_before = re.escape(stripped_context_before)

    match = re.search(escaped_context_before, stripped_cleaned_response)
    if match:
        completion_start = match.end()
    else:
        fallback_context = stripped_context_before[-30:] if len(stripped_context_before) > 30 else stripped_context_before
        escaped_fallback = re.escape(fallback_context)
        
        match = re.search(escaped_fallback, stripped_cleaned_response)
        if match:
            completion_start = match.end()
        else:
            completion = stripped_cleaned_response
            if len(completion) > 500:
                return "█"
            return completion
    
    after_pos = -1
    if context_after.strip():
        stripped_context_after = context_after.strip()
        escaped_context_after = re.escape(stripped_context_after)

        after_match = re.search(escaped_context_after, stripped_cleaned_response[completion_start:])
        if after_match:
            after_pos = completion_start + after_match.start()
        else:
            fallback_after_context = stripped_context_after[:30] if len(stripped_context_after) > 30 else stripped_context_after
            escaped_fallback_after = re.escape(fallback_after_context)
            
            after_match = re.search(escaped_fallback_after, stripped_cleaned_response[completion_start:])
            if after_match:
                after_pos = completion_start + after_match.start()
    
    if after_pos != -1:
        completion = stripped_cleaned_response[completion_start:after_pos]
    else:
        completion = stripped_cleaned_response[completion_start:]
        
        if len(completion.strip()) > 2000:
            return "█"

    completion = completion.strip()
    
    if not completion:
        return "█"

    if len(completion) > 1000:
        return "█"

    return completion

# Test cases in tabular format
TEST_CASES = [
    # Basic single line completion
    {
        "before_context": "def hello():",
        "after_context": "",
        "llm_response": """def hello():
    print('Hello World')""",
        "expected_extraction": "print('Hello World')"
    },
    
    # Multi-line completion
    {
        "before_context": "if condition:",
        "after_context": """
else:""",
        "llm_response": """if condition:
    x = 1
    y = 2
else:""",
        "expected_extraction": """x = 1
    y = 2"""
    },
    
    # With whitespace preservation
    {
        "before_context": "    def func():",
        "after_context": "",
        "llm_response": """    def func():
        return 42""",
        "expected_extraction": "return 42"
    },
    
    # LLM reproduces full context
    {
        "before_context": """print('start')
for i in range(10):""",
        "after_context": """
print('end')""",
        "llm_response": """print('start')
for i in range(10):
    print(i)
print('end')""",
        "expected_extraction": "print(i)"
    },
    
    # LLM doesn't reproduce after context
    {
        "before_context": "x = ",
        "after_context": "",
        "llm_response": "x = 5 + 3",
        "expected_extraction": "5 + 3"
    },
    
    # 30-character before fallback
    {
        "before_context": "This is a very long line of code that exceeds thirty characters and should trigger fallback",
        "after_context": "",
        "llm_response": """thirty characters and should trigger fallback
    new_code_here""",
        "expected_extraction": "new_code_here"
    },
    
    # 30-character after fallback  
    {
        "before_context": "result = ",
        "after_context": "# This is a very long comment that exceeds thirty characters",
        "llm_response": "result = calculate_value()# This is a very long comment that",
        "expected_extraction": "calculate_value()"
    },
    
    # Markdown code block removal
    {
        "before_context": "def test():",
        "after_context": "",
        "llm_response": """```python
def test():
    return True
```""",
        "expected_extraction": "return True"
    },
    
    # Malformed markdown (no closing ```)
    {
        "before_context": "class MyClass:",
        "after_context": "",
        "llm_response": """```python
class MyClass:
    def __init__(self):
        pass""",
        "expected_extraction": """def __init__(self):
        pass"""
    },
    
    # Empty completion
    {
        "before_context": "print()",
        "after_context": "",
        "llm_response": "print()",
        "expected_extraction": "█"
    },
    
    # Too long completion (over 1000 chars)
    {
        "before_context": "start",
        "after_context": "",
        "llm_response": f"start{'x' * 1001}",
        "expected_extraction": "█"
    },
    
    # No before context match - fallback to whole response
    {
        "before_context": "nonexistent",
        "after_context": "",
        "llm_response": "some_completion_code",
        "expected_extraction": "some_completion_code"
    },
    
    # No before context match - too long fallback
    {
        "before_context": "nonexistent",
        "after_context": "",
        "llm_response": "x" * 501,
        "expected_extraction": "█"
    },
    
    # Whitespace handling
    {
        "before_context": "  if True:",
        "after_context": "  else:",
        "llm_response": """  if True:
    pass
  else:""",
        "expected_extraction": "pass"
    },
    
    # Special regex characters in context
    {
        "before_context": "x = re.compile(r'\\d+')",
        "after_context": "",
        "llm_response": """x = re.compile(r'\\d+')
y = x.match('123')""",
        "expected_extraction": "y = x.match('123')"
    },
    
    # Multiple occurrences of before context
    {
        "before_context": "print",
        "after_context": "",
        "llm_response": """print('first')
print('second')""",
        "expected_extraction": """('first')
print('second')"""
    },
    
    # Complex multiline with indentation
    {
        "before_context": "try:",
        "after_context": "except:",
        "llm_response": """try:
    file = open('test.txt')
    content = file.read()
except:""",
        "expected_extraction": """file = open('test.txt')
    content = file.read()"""
    },
    
    # Only after context available
    {
        "before_context": "",
        "after_context": "return result",
        "llm_response": """calculate_something()
return result""",
        "expected_extraction": "calculate_something()"
    },
    
    # Before context with trailing whitespace
    {
        "before_context": "def func():  ",
        "after_context": "",
        "llm_response": """def func():
    return None""",
        "expected_extraction": "return None"
    },
    
    # After context with leading whitespace
    {
        "before_context": "x = ",
        "after_context": "  # comment",
        "llm_response": "x = 42  # comment",
        "expected_extraction": "42"
    },
    
    # Mixed tabs and spaces
    {
        "before_context": "def test():",
        "after_context": "\treturn",
        "llm_response": """def test():
\tprint('hello')
\treturn""",
        "expected_extraction": "print('hello')"
    },
    
    # Unicode characters
    {
        "before_context": "message = ",
        "after_context": "",
        "llm_response": "message = 'Hello 世界'",
        "expected_extraction": "'Hello 世界'"
    },
    
    # Empty before and after context
    {
        "before_context": "",
        "after_context": "",
        "llm_response": "standalone_code",
        "expected_extraction": "standalone_code"
    },
    
    # Very short completion
    {
        "before_context": "x = ",
        "after_context": "",
        "llm_response": "x = 1",
        "expected_extraction": "1"
    },
    
    # Completion with newlines at end
    {
        "before_context": "def test():",
        "after_context": "",
        "llm_response": """def test():
    print('test')


""",
        "expected_extraction": "print('test')"
    },
    
    # No placeholder in context (should return empty)
    {
        "before_context": "normal text",
        "after_context": "",
        "llm_response": "normal text completion",
        "expected_extraction": ""
    },
    
    # LLM adds extra content before expected context
    {
        "before_context": "result = calculate()",
        "after_context": "",
        "llm_response": """# Some comment
result = calculate()
print(result)""",
        "expected_extraction": "print(result)"
    },
    
    # Exact boundary case - 30 chars
    {
        "before_context": "exactly_thirty_characters_her",
        "after_context": "",
        "llm_response": "exactly_thirty_characters_hercompletion",
        "expected_extraction": "completion"
    },
    
    # Case where LLM response is shorter than expected
    {
        "before_context": "very long context that won't be found",
        "after_context": "",
        "llm_response": "short",
        "expected_extraction": "short"
    },
]

def run_tests():
    print("Testing inline completion extraction logic\n")
    print(f"{'#':<3} {'Before Context':<40} {'After Context':<20} {'Expected':<20} {'Actual':<20} {'Pass':<5}")
    print("-" * 110)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        before_context = test_case["before_context"]
        after_context = test_case["after_context"]
        llm_response = test_case["llm_response"]
        expected = test_case["expected_extraction"]
        
        # Construct original context with placeholder
        original_context = f"{before_context}█{after_context}"
        
        # Run extraction
        actual = extract_completion_from_response(llm_response, original_context, 0)
        
        # Check result
        is_pass = actual == expected
        if is_pass:
            passed += 1
        else:
            failed += 1
        
        # Format for display (truncate long strings)
        before_display = before_context[:37] + "..." if len(before_context) > 40 else before_context
        after_display = after_context[:17] + "..." if len(after_context) > 20 else after_context
        expected_display = expected[:17] + "..." if len(expected) > 20 else expected
        actual_display = actual[:17] + "..." if len(actual) > 20 else actual
        
        status = "✓" if is_pass else "✗"
        
        print(f"{i:<3} {before_display:<40} {after_display:<20} {expected_display:<20} {actual_display:<20} {status:<5}")
    
    print("-" * 110)
    print(f"\nResults: {passed} passed, {failed} failed, {len(TEST_CASES)} total")
    
    if failed > 0:
        print("\nFailed test details:")
        for i, test_case in enumerate(TEST_CASES, 1):
            before_context = test_case["before_context"]
            after_context = test_case["after_context"]
            llm_response = test_case["llm_response"]
            expected = test_case["expected_extraction"]
            
            original_context = f"{before_context}█{after_context}"
            actual = extract_completion_from_response(llm_response, original_context, 0)
            if actual != expected:
                print(f"\nTest {i}:")
                print(f"  Before: {repr(before_context)}")
                print(f"  After:  {repr(after_context)}")
                print(f"  LLM:    {repr(llm_response)}")
                print(f"  Expected: {repr(expected)}")
                print(f"  Actual:   {repr(actual)}")

if __name__ == "__main__":
    run_tests()
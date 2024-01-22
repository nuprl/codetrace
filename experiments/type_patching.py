"""
Most basic experiment:
clean prompt - corrupted prompt pairs
Can we recover the clean prompt from the corrupted prompt?

Idea:
clean prompt is predict type X FIM - corrupted prompt is predict type Y FIM
where X != Y

We can assume starcoder will guess all primitive types as we have seen in tests
"""
# TypeScript types
basic_types = [
    "string",
    "number",
    "boolean",
]

# collect all snippets containing the target type annotations
# break at function definitions
# ensure tokenization length is same all across

def collect_snippets():
    pass

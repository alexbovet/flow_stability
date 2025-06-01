"""Unsorted collection of simple helper functions
"""

def include_doc_from(cls):
    """Decorator to include the docstring from a specified class."""
    def decorator(func):
        func.__doc__ += "\n" + (cls.__doc__ or "")
        return func
    return decorator

def inverted_iterator(iterator):
    """Create an inverted iterator
    """
    for val in iterator:
        if val is None:
            yield val
        else:
            yield 1. / val

import pytest
from flowstab.helpers import include_doc_from, inverted_iterator

def test_include_doc_from_merges_docstrings():
    class Dummy:
        """This is Dummy's docstring."""

    @include_doc_from(Dummy)
    def some_func():
        """Func doc."""
        pass

    # The resulting docstring should include both
    assert "Func doc." in some_func.__doc__
    assert "This is Dummy's docstring." in some_func.__doc__

def test_inverted_iterator_regular_values():
    data = [2, 4, 0.5]
    result = list(inverted_iterator(data))
    assert result == [0.5, 0.25, 2.0]

def test_inverted_iterator_with_none():
    data = [2, None, 4]
    result = list(inverted_iterator(data))
    assert result == [0.5, None, 0.25]

def test_inverted_iterator_empty():
    assert list(inverted_iterator([])) == []

def test_inverted_iterator_raises_on_zero():
    # Check if ZeroDivisionError is raised when 0 is in the iterator
    data = [0]
    with pytest.raises(ZeroDivisionError):
        list(inverted_iterator(data))

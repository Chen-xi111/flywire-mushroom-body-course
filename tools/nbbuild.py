"""Tiny helpers to write each lesson once and build two notebooks from it.

A lesson module defines ``CELLS``: a list made with the helpers below.

* ``md(text)``      -> the same markdown cell in both versions
* ``code(src)``     -> the same code cell in both versions (provided code)
* ``todo(ex, sol)`` -> a code cell: ``ex`` in the exercise, ``sol`` in the solution
* ``answer(ex, sol)`` -> a markdown cell: blank answer template vs. model answer

Run ``python tools/build.py`` from the repository root to regenerate
``notebooks/`` (exercises) and ``solutions/`` (worked solutions).
"""
from textwrap import dedent


def _clean(s):
    return dedent(s).strip("\n")


def md(text):
    return ("md", _clean(text), _clean(text))


def code(src):
    return ("code", _clean(src), _clean(src))


def todo(exercise, solution):
    return ("code", _clean(exercise), _clean(solution))


def answer(exercise, solution):
    return ("md", _clean(exercise), _clean(solution))

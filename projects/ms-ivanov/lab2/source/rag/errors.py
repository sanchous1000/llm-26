"""Errors that are shown to the user as plain messages."""


class RagError(Exception):
    """The question could not be answered (index missing, LLM failure, ...)."""

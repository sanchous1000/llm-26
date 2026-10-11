"""Exceptions raised by the network layer."""


class FetchError(Exception):
    """A resource could not be fetched.

    ``permanent`` tells whether retrying later makes sense: a 404 or a body
    that is not the expected document type is permanent, a timeout or a
    5xx answer is not.
    """

    def __init__(self, message: str, *, permanent: bool, status: int | None = None):
        super().__init__(message)
        self.permanent = permanent
        self.status = status

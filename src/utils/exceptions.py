class CustomException(Exception):
    """
    Custom exception class for handling project-specific errors.
    """

    def __init__(self, message: str, errors: Exception = None):
        super().__init__(message)
        self.errors = errors

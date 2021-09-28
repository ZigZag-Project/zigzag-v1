class OutputNodeOverfullException(Exception):
    """Exception raised for an output memory level that is too full.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, message="Output Node can't hold all loops from level below"):

        self.message = message
        super().__init__(self.message)

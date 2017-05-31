class InvalidToeplitzException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return repr(self.message)


class InvalidToeplitzBlockSize(InvalidToeplitzException):
    def __init__(self, N, m):
        self.m = m
        self.N = N
        self.message = "Size of matrix N = {} is not an integer multiple of block size n = {}".format(self.N, self.m)


class InvalidMethodException(Exception):
    def __init__(self, method):
        self.message = "{} is not a valid method".format(method)
    def __str__(self):
        return repr(self.message)

class InvalidPException(Exception):
    def __init__(self, p):
        self.message = "p = {} is not greator or equal to 1".format(p)
    def __str__(self):
        return repr(self.message)
        

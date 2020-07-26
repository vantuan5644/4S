from enum import Enum


class ModelException(Exception):
    def __(self, code, message):
        self.code = code
        self.message = message


class ExceptionType(Enum):
    INPUT_ERROR = 100
    REGISTRATION_ERROR = 101
    VERIFICATION_ERROR = 102
    FACE_DETECTION_ERROR = 103

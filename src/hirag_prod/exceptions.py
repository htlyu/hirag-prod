class HiRAGException(Exception):
    """HiRAG base exception class"""


class DocumentProcessingError(HiRAGException):
    """Document processing exception"""


class KGConstructionError(HiRAGException):
    """Knowledge graph construction exception"""


class StorageError(HiRAGException):
    """Storage exception"""

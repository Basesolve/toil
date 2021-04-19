from typing import Any

class CryptoError(Exception): ...
class BadSignatureError(CryptoError): ...
class RuntimeError(RuntimeError, CryptoError): ...
class AssertionError(AssertionError, CryptoError): ...
class TypeError(TypeError, CryptoError): ...
class ValueError(ValueError, CryptoError): ...
class InvalidkeyError(CryptoError): ...

def ensure(cond: Any, *args: Any, **kwds: Any) -> None: ...
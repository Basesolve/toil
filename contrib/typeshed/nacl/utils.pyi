from typing import Any

class EncryptedMessage(bytes):
    @property
    def nonce(self): ...
    @property
    def ciphertext(self): ...

class StringFixer: ...

def bytes_as_string(bytes_in: Any): ...
def random(size: int = ...): ...
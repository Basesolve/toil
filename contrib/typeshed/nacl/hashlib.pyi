from nacl.utils import bytes_as_string as bytes_as_string
from typing import Any

BYTES: Any
BYTES_MIN: Any
BYTES_MAX: Any
KEYBYTES: Any
KEYBYTES_MIN: Any
KEYBYTES_MAX: Any
SALTBYTES: Any
PERSONALBYTES: Any

class blake2b:
    MAX_DIGEST_SIZE: Any = ...
    MAX_KEY_SIZE: Any = ...
    PERSON_SIZE: Any = ...
    SALT_SIZE: Any = ...
    def __init__(self, data: bytes = ..., digest_size: Any = ..., key: bytes = ..., salt: bytes = ..., person: bytes = ...) -> None: ...
    @property
    def digest_size(self): ...
    @property
    def block_size(self): ...
    @property
    def name(self): ...
    def update(self, data: Any) -> None: ...
    def digest(self): ...
    def hexdigest(self): ...
    def copy(self): ...

def scrypt(password: Any, salt: str = ..., n: Any = ..., r: int = ..., p: int = ..., maxmem: Any = ..., dklen: int = ...): ...
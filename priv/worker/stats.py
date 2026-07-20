"""Worker statistics counters."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Stats:
    compile_count: int = 0
    execute_count: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    duplicate_input_clones: int = 0
    immutable_copy_fast: int = 0
    immutable_copy_fallback: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

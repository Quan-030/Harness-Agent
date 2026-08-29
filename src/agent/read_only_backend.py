# src/agent/read_only_backend.py
"""Read-only StoreBackend for the shared system-skills route.

The /persisted-skills/system/ route is backed by the shared
("system_skills",) namespace. Only SkillsSyncMiddleware may populate it (it
writes src/skills/ via store.put directly, bypassing this backend). An agent
session must NOT be able to write into the shared namespace — otherwise a
user's skill would leak to every user, breaking "only src/skills is shared".

This subclass keeps every read (read/ls/grep/glob/download) but turns each
mutation into an error result the agent sees as a normal tool failure. All six
mutating entry points are overridden explicitly: StoreBackend.awrite/aedit have
real async implementations (they do NOT delegate to the sync versions), so
overriding only the sync methods would leave the async path writable.
"""
from __future__ import annotations

from deepagents.backends.protocol import (
    EditResult,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.store import StoreBackend

_READ_ONLY_MSG = (
    "系统技能目录 /persisted-skills/system/ 为只读，无法写入。"
    "如需持久化自己的技能，请写入 /persisted-skills/（个人空间）或使用 assign_skill 工具。"
)


class ReadOnlyStoreBackend(StoreBackend):
    """StoreBackend that rejects all writes; reads delegate to StoreBackend."""

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_READ_ONLY_MSG)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_READ_ONLY_MSG)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(error=_READ_ONLY_MSG)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(error=_READ_ONLY_MSG)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]

"""IPC wire protocol: JSON message types, encode/decode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class MsgType(str, Enum):
    """IPC message types."""

    # Requests (client -> server)
    SUBMIT = "submit"
    CANCEL = "cancel"
    STATUS = "status"
    RESULT = "result"
    HEALTH = "health"
    # Responses (server -> client)
    SUBMIT_ACK = "submit_ack"
    CANCEL_ACK = "cancel_ack"
    STATUS_RESP = "status_resp"
    RESULT_RESP = "result_resp"
    HEALTH_RESP = "health_resp"


@dataclass(slots=True)
class ArtifactRef:
    """Reference to a completed artifact on tmpfs."""

    path: str
    content_type: str = "application/x-npy"
    size_bytes: int = 0
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "path": self.path,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
        }
        if self.shape is not None:
            d["shape"] = list(self.shape)
        if self.dtype is not None:
            d["dtype"] = self.dtype
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ArtifactRef:
        shape = tuple(d["shape"]) if d.get("shape") is not None else None
        return cls(
            path=d["path"],
            content_type=d.get("content_type", "application/x-npy"),
            size_bytes=d.get("size_bytes", 0),
            shape=shape,
            dtype=d.get("dtype"),
        )


def encode_msg(msg_type: str, cid: str, payload: dict) -> bytes:
    msg = {"type": msg_type, "cid": cid, **payload}
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def decode_msg(raw: bytes) -> tuple[str, str, dict]:
    msg = json.loads(raw)
    msg_type = msg.pop("type")
    cid = msg.pop("cid")
    return msg_type, cid, msg

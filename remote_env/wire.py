"""Wire format shared by the remote Isaac Lab env server and client.

Tensors travel as safetensors (no pickle, no code execution); scalars/strings go in a JSON blob stored
in the safetensors metadata.
"""

import json

import grpc
import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

SERVICE = "isaacenv.Env"
METHODS = ("Init", "Reset", "GetObs", "Step", "SetEpisodeLen", "Close")
MAX_MSG = 1 << 30
CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MSG),
    ("grpc.max_receive_message_length", MAX_MSG),
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_permit_without_calls", 1),
]


def pack(tensors: dict[str, torch.Tensor] | None = None, meta: dict | None = None) -> bytes:
    t = {k: v.detach().contiguous().cpu() for k, v in (tensors or {}).items()}
    return st_save(t, metadata={"m": json.dumps(meta or {})})


def unpack(buf: bytes) -> tuple[dict[str, torch.Tensor], dict]:
    n = int.from_bytes(buf[:8], "little")
    header = json.loads(buf[8 : 8 + n])
    meta = json.loads(header.get("__metadata__", {}).get("m", "{}"))
    return st_load(buf), meta


def make_server_handler(dispatch, token: str):
    """dispatch(method, request_bytes) -> response_bytes; called from gRPC worker threads."""
    import hmac

    def wrap(name):
        def fn(request: bytes, context):
            md = dict(context.invocation_metadata())
            if not hmac.compare_digest(md.get("x-token", ""), token):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad token")
            return dispatch(name, request)

        return grpc.unary_unary_rpc_method_handler(fn, request_deserializer=None, response_serializer=None)

    return grpc.method_handlers_generic_handler(SERVICE, {m: wrap(m) for m in METHODS})

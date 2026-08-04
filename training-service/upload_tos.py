"""Upload training artifacts to Volcengine TOS. A training job calls this when it finishes to
push checkpoints and logs off the (diskless) pod.

Credentials come from environment variables provided by a K8s Secret mounted into the job;
nothing is hard-coded:
  TOS_ACCESS_KEY / TOS_SECRET_KEY / TOS_BUCKET
  TOS_ENDPOINT (defaults to the Beijing internal endpoint) / TOS_REGION (defaults to cn-beijing)

Usage:
  python upload_tos.py --prefix <run-name> [--local-dir /workspace/isaaclab/logs]
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload training artifacts to TOS")
    ap.add_argument("--local-dir", default="/workspace/isaaclab/logs", help="local directory to upload")
    ap.add_argument("--prefix", required=True, help="object prefix on TOS (typically the output_name or job name)")
    args = ap.parse_args()

    try:
        ak = os.environ["TOS_ACCESS_KEY"]
        sk = os.environ["TOS_SECRET_KEY"]
        bucket = os.environ["TOS_BUCKET"]
    except KeyError as e:
        print(f"[tos] missing environment variable {e}; skipping upload (training finished, artifacts are still in the container at {args.local_dir})", file=sys.stderr)
        return 0  # missing credentials must not fail the whole job

    endpoint = os.environ.get("TOS_ENDPOINT", "tos-cn-beijing.ivolces.com")
    region = os.environ.get("TOS_REGION", "cn-beijing")

    if not os.path.isdir(args.local_dir):
        print(f"[tos] local directory does not exist, nothing to upload: {args.local_dir}")
        return 0

    import tos  # Volcengine TOS python SDK, installed in the image

    client = tos.TosClientV2(ak, sk, endpoint, region)
    n, total = 0, 0
    for root, _, files in os.walk(args.local_dir):
        for f in files:
            fp = os.path.join(root, f)
            key = f"{args.prefix.strip('/')}/{os.path.relpath(fp, args.local_dir)}"
            client.put_object_from_file(bucket, key, fp)
            n += 1
            total += os.path.getsize(fp)
            print(f"[tos] {fp} -> tos://{bucket}/{key}", flush=True)

    print(f"[tos] done: {n} files / {total/1e6:.1f} MB -> tos://{bucket}/{args.prefix.strip('/')}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

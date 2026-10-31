#!/usr/bin/env python3
"""Wait until a service accepts TCP connections: wait_for.py <url> <default port> <seconds>."""

import socket
import sys
import time
from urllib.parse import urlparse


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: wait_for.py <url> <default port> <seconds>")
        return 1
    url, default_port, seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port or default_port
    if not host:
        print(f"ERROR: no host in {parsed.scheme}:// URL")
        return 1
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return 0
        except OSError:
            time.sleep(1)
    print(f"ERROR: {host}:{port} not reachable after {seconds:.0f} s")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

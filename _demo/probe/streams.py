"""Send zeros to <host> <port> over 1, 8 and 32 TCP streams for 10 s each and report the rate."""
import socket
import sys
import threading
import time

host, port = sys.argv[1], int(sys.argv[2])
chunk = bytes(1 << 20)


def run(streams: int, seconds: float = 10.0) -> None:
    sent = [0] * streams
    deadline = time.monotonic() + seconds

    def send(i: int) -> None:
        with socket.create_connection((host, port)) as conn:
            while time.monotonic() < deadline:
                sent[i] += conn.send(chunk)

    start = time.monotonic()
    threads = [threading.Thread(target=send, args=(i,)) for i in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - start
    print(f"{streams} stream(s): {sum(sent) * 8 / elapsed / 1e9:.3f} Gbit/s ({sum(sent) / 2**20:.0f} MiB in {elapsed:.1f}s)")


for streams in (1, 8, 32):
    run(streams)

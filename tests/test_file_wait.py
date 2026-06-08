"""Tests for wait_for_file and wait_for_finished_file."""

import threading
import time
from pathlib import Path

import pytest

from xrpd_toolbox.utils.utils import wait_for_file, wait_for_finished_file


def create_file_after(path: Path, delay: float, content: bytes = b"") -> None:
    def _worker():
        time.sleep(delay)
        path.write_bytes(content)

    threading.Thread(target=_worker, daemon=True).start()


def grow_file_after(path: Path, chunks: list[bytes], chunk_interval: float) -> None:
    def _worker():
        path.write_bytes(chunks[0])
        for chunk in chunks[1:]:
            time.sleep(chunk_interval)
            with path.open("ab") as fh:
                fh.write(chunk)

    threading.Thread(target=_worker, daemon=True).start()


def test_wait_for_file_already_exists(tmp_path):
    f = tmp_path / "ready.txt"
    f.write_bytes(b"")
    assert wait_for_file(f) is None


def test_wait_for_file_appears_before_timeout(tmp_path):
    f = tmp_path / "late.txt"
    create_file_after(f, delay=0.2)
    wait_for_file(f, timeout=2, poll_interval=0.05)


def test_wait_for_file_timeout(tmp_path):
    f = tmp_path / "missing.txt"
    with pytest.raises(TimeoutError, match=r"does not exist after 0.2"):
        wait_for_file(f, timeout=0.2, poll_interval=0.05)


def test_wait_for_file_no_timeout(tmp_path):
    f = tmp_path / "eventual.txt"
    create_file_after(f, delay=0.2)
    wait_for_file(f, timeout=None, poll_interval=0.05)


def test_wait_for_file_accepts_string_path(tmp_path):
    f = tmp_path / "str.txt"
    f.write_bytes(b"")
    wait_for_file(str(f))


def test_wait_for_finished_file_already_stable(tmp_path):
    f = tmp_path / "stable.txt"
    f.write_bytes(b"done")
    assert wait_for_finished_file(f, stable_for=0.1, poll_interval=0.05) is None


def test_wait_for_finished_file_waits_while_growing(tmp_path):
    f = tmp_path / "growing.txt"
    grow_file_after(f, chunks=[b"part1", b"part2", b"part3"], chunk_interval=0.15)
    wait_for_finished_file(f, timeout=5, stable_for=0.3, poll_interval=0.05)


def test_wait_for_finished_file_timeout_missing(tmp_path):
    f = tmp_path / "ghost.txt"
    with pytest.raises(TimeoutError, match="does not exist"):
        wait_for_finished_file(f, timeout=0.2, stable_for=0.1, poll_interval=0.05)


def test_wait_for_finished_file_timeout_still_writing(tmp_path):
    f = tmp_path / "infinite.txt"

    def _keep_writing():
        f.write_bytes(b"x")
        for _ in range(20):
            time.sleep(0.05)
            with f.open("ab") as fh:
                fh.write(b"x")

    threading.Thread(target=_keep_writing, daemon=True).start()

    with pytest.raises(TimeoutError, match="did not finish writing"):
        wait_for_finished_file(f, timeout=0.4, stable_for=1.0, poll_interval=0.05)


def test_wait_for_finished_file_no_timeout(tmp_path):
    f = tmp_path / "slow.txt"
    create_file_after(f, delay=0.2, content=b"finally")
    wait_for_finished_file(f, timeout=None, stable_for=0.15, poll_interval=0.05)


def test_wait_for_finished_file_accepts_string_path(tmp_path):
    f = tmp_path / "str.txt"
    f.write_bytes(b"done")
    wait_for_finished_file(str(f), stable_for=0.1, poll_interval=0.05)

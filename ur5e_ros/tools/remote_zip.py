"""Doc / giai nen tung file trong 1 file zip tren HTTP bang range request (khong tai ca zip).

    python tools/remote_zip.py list <url>
    python tools/remote_zip.py get  <url> <ten_trong_zip> [<ten> ...] --out <thu_muc>
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


class HttpRangeFile(io.RawIOBase):
    """File-like chi doc, moi lan read() la 1 HTTP range request."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()  # theo redirect (Hugging Face -> CDN)
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        self.downloaded += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url):
    f = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 18)), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "get"])
    ap.add_argument("url")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    zf, raw = open_remote_zip(args.url)
    if args.cmd == "list":
        for info in zf.infolist():
            print(f"{info.file_size:>12d}  {info.filename}")
    else:
        out = Path(args.out)
        for name in args.names:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src:
                target.write_bytes(src.read())
            print(f"OK {name} -> {target} ({target.stat().st_size} B)")
    print(f"# zip {raw.size / 1e6:.1f} MB, da tai {raw.downloaded / 1e6:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()

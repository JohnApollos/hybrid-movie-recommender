"""Download & extract MovieLens ml-latest-small. Run: python data/download.py"""
import io, urllib.request, zipfile, pathlib

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
dest = pathlib.Path(__file__).parent
print("Downloading ml-latest-small (~1 MB)...")
with urllib.request.urlopen(URL, timeout=60) as r:
    zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
print("Done ->", dest / "ml-latest-small")

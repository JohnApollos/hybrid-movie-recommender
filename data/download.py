"""Download & extract MovieLens ml-latest-small. Run: python data/download.py"""
import certifi, io, pathlib, ssl, urllib.request, zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
dest = pathlib.Path(__file__).parent
print("Downloading ml-latest-small (~1 MB)...")
context = ssl.create_default_context(cafile=certifi.where())
with urllib.request.urlopen(URL, timeout=60, context=context) as r:
    zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
print("Done ->", dest / "ml-latest-small")

"""Download and extract MovieLens ml-latest-small."""
import certifi
import io
import pathlib
import ssl
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_dataset(destination: pathlib.Path | None = None) -> pathlib.Path:
    """Download the MovieLens archive when the local metadata is unavailable."""
    destination = destination or pathlib.Path(__file__).parent
    dataset_dir = destination / "ml-latest-small"
    if all((dataset_dir / filename).is_file() for filename in ("movies.csv", "ratings.csv", "links.csv")):
        return dataset_dir

    destination.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, timeout=60, context=context) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        bundle.extractall(destination)
    if not all((dataset_dir / filename).is_file() for filename in ("movies.csv", "ratings.csv", "links.csv")):
        raise RuntimeError(f"MovieLens archive did not contain the expected files in {dataset_dir}.")
    return dataset_dir


if __name__ == "__main__":
    print("Downloading ml-latest-small (~1 MB)...")
    print("Done ->", download_dataset())

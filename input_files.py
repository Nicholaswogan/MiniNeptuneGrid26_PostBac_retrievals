import warnings
import os
warnings.filterwarnings('ignore')
THISFILE = os.path.dirname(os.path.abspath(__file__))
os.environ["picaso_refdata"] = os.path.join(THISFILE, "picasofiles", "reference")
os.environ["PYSYN_CDBS"] = os.path.join(
    THISFILE, 
    "picasofiles", 
    "reference", 
    "stellar_grids",
    "grp",
    "redcat",
    "trds",
)

import urllib.parse
import urllib.request
import tarfile
import zipfile
from picaso import data

def download_and_extract_archive(archive_url, destination_folder, archive_filename=None, delete_archive=False, show_progress=True):
    os.makedirs(destination_folder, exist_ok=True)

    if archive_filename is None:
        archive_filename = os.path.basename(urllib.parse.urlparse(archive_url).path) or "download.zip"

    archive_path = os.path.join(destination_folder, archive_filename)

    def _progress_hook(block_num, block_size, total_size):
        if not show_progress:
            return
        downloaded = block_num * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
            percent = (downloaded / total_size) * 100.0
            print(f"\rDownloading {archive_filename}: {percent:6.2f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
        else:
            print(f"\rDownloading {archive_filename}: {downloaded} bytes", end="", flush=True)

    urllib.request.urlretrieve(archive_url, archive_path, reporthook=_progress_hook)
    if show_progress:
        print()

    if archive_filename.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination_folder)
    elif archive_filename.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination_folder)
    else:
        raise ValueError(f"Unsupported archive format: {archive_filename}")

    if delete_archive:
        os.remove(archive_path)

    return archive_path


def main():

    if not os.path.exists('picasofiles/opacities_photochem_0.1_250.0_R15000_v2.db'):
        url = "https://zenodo.org/records/20397663/files/opacities_photochem_0.1_250.0_R15000_v2.db.zip"
        download_and_extract_archive(
            archive_url=url,
            destination_folder='picasofiles', 
            delete_archive=True,
        )
    else:
        print('Opacity files is already downloaded')


    stellar_grid_marker = os.path.join(
        'picasofiles',
        'reference',
        'stellar_grids',
        'grp',
    )
    if not os.path.exists(stellar_grid_marker):
        url = "http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz"
        download_and_extract_archive(
            archive_url=url,
            destination_folder=os.path.join('picasofiles', 'reference', 'stellar_grids'),
            delete_archive=True,
        )
    else:
        print('Stellar grids are already downloaded')


if __name__ == '__main__':
    main()

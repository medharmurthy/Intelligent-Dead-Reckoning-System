import os
import urllib.request
import urllib.parse

BASE_LFS_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/"

def download_file(relative_path, target_filepath):
    # Construct the LFS media URL
    # Replace backslashes with forward slashes and URL encode path segments
    parts = relative_path.replace("\\", "/").split("/")
    encoded_parts = [urllib.parse.quote(part) for part in parts]
    full_url = BASE_LFS_URL + "/".join(encoded_parts)
    
    print(f"Downloading: {relative_path}")
    print(f"From URL:    {full_url}")
    
    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
    
    # Download with progress report
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"\rProgress: {downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB ({percent:.1f}%)", end="")
        else:
            print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="")
            
    urllib.request.urlretrieve(full_url, target_filepath, reporthook=report_progress)
    print("\nDownload complete!\n")

if __name__ == "__main__":
    # Download sample trip S1 for initial baseline exploration
    trip_rel_dir = "Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1"
    
    files_to_download = [
        f"{trip_rel_dir}/S-S1.csv",
        f"{trip_rel_dir}/V-S1.csv"
    ]
    
    base_target = "data/raw/IO-VNBD/IO-VNBD-master"
    
    for rel_path in files_to_download:
        target_path = os.path.join(base_target, rel_path)
        download_file(rel_path, target_path)

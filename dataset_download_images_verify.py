import argparse
import os
import requests
import pandas as pd
from tqdm import tqdm
import hashlib
from PIL import Image
import cv2
import pdqhash

err = open("errors.txt",'w')


def compute_pdqhash(image_path):
    global err
    """
    Reads an image file, computes its PDQ hash, and writes the hash to a folder.

    Args:
        image_path (str): Path to the image file.
        output_folder (str): Path to the folder where the hash will be saved.
    """
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' does not exist.", file=err)
        return
    image = cv2.imread(image_path)
    if image is None:
        print(image_path, "is not an image", file=err)
        return ""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hash_value, quality = pdqhash.compute(image)
    # Convert the hash value to a binary string
    hash_value = [str(int(bit)) for bit in hash_value]
    hex_hash = str(hex(int(''.join(hash_value), 2))[2:]).zfill(64)
    return hex_hash

def compute_pdqhash_distance(hash1, hash2):
    global err
    """
    Computes the PDQ hash distance between two hash values.

    Args:
        hash1 (str): First hash value.
        hash2 (str): Second hash value.

    Returns:
        int: The distance between the two hashes.
    """
    hash1 = int("0x" + hash1, 16)
    hash1 = format(hash1, '#0256b')[2:].zfill(256)

    hash2 = int("0x" + hash2, 16)
    hash2 = format(hash2, '#0256b')[2:].zfill(256)

    if len(hash1) != len(hash2):
        raise ValueError("Hash values must be of the same length.", file=err)
    return sum(h1 != h2 for h1, h2 in zip(hash1, hash2))

def is_valid_jpg(file_path):
    global err
    """Check if a file is a valid JPEG image."""
    try:
        with Image.open(file_path) as img:
            if img.format != 'JPEG':
                print(f"File is {img.format}, and is not saved", file=err)
                return False
            else:
                return True
    except Exception as e:
        print(f"File '{file_path}' is not a valid JPEG: {e}", file=err)
        return False
    

def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def is_image_url(url):
    global err
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0'}
    
    if url.endswith(".jpg") or url.endswith(".jpeg")  or url.endswith(".JPEG") or url.endswith(".JPG"):
        return True # if url is not a jpg, it will be checked in the main function

    try:
        # Send a HEAD request to check the URL
        response = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        response.raise_for_status()
        # Check if the Content-Type is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith("image"):
            print(content_type, file=err)
        return content_type.startswith('image')
    except Exception as e:
        print(f"URL check failed for {url}: {e}", file=err)
        return False

def download_image(url, folder, image_name=None):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0'}

    response = requests.get(url, stream=True, timeout=10, headers=headers)
    response.raise_for_status()
        # Extract the filename from the URL
    if image_name is None:
        filename = os.path.basename(url)
    else:
        filename = image_name
        
    file_path = os.path.join(folder, filename)
        # Save the image to the folder
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)
    #print(f"Downloaded: {filename}")

def main():
    global err
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create a folder and download images from URLs.")
    
    # Add arguments
    parser.add_argument('--folder', type=str, required=True, help="Name of the folder to create.")
    parser.add_argument('--dataframe', type=str, default="image_caption_children_in_the_wild_dataset.csv", help="Path to a dataframe file containing the dataset.")
    parser.add_argument('--force', action='store_true', help="Force download images even if the folder already exists.")
    parser.add_argument('--verify-only', action='store_true', help="Verify only without downloading data")
    parser.add_argument('--keep-non-verified', action='store_true', help="Keep files with PDQ hash distance larger than the threshold")
    parser.add_argument('--pdq-threshold', type=int, default=8, help="Threshold for Hamming distance between PDQ hashes")

    # Parse arguments
    args = parser.parse_args()
    
    # Get the folder name and URLs file path
    folder_name = args.folder
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    url_fails = 0
    # Read URLs from the file and download images
    if os.path.exists(args.dataframe):
        df = pd.read_csv(args.dataframe)
    else:
        print(f"File '{args.dataframe}' does not exist. Please check the path.")
    

    print("\nIMAGE DOWNLOAD:", file=err)
    urls = df['url'].tolist()
    if not args.verify_only:
        for url in tqdm(urls):
            #image_format = url.split('.')[-1]
            image_format = "jpg"
            image_path = os.path.join(folder_name, "%d.%s" % (urls.index(url), image_format))
            if args.force or not os.path.exists(image_path):
                url = url.strip()  # Remove any leading/trailing whitespace
                if url:
                    if is_image_url(url):
                        try:
                            download_image(url, folder_name, image_name="%d.%s" % (urls.index(url), image_format))
                            if not is_valid_jpg(image_path):
                                print(f"Downloaded image from URL={url} to file '{image_path}', but the file is not a valid JPEG image. We remove the file.", file=err)
                                url_fails += 1
                                if not args.keep_non_verified:
                                    os.remove(image_path)
                        except Exception as e:
                            print(f"Failed to download image from URL={url}: {e}", file=err)
                            url_fails += 1
                    else:
                        print(f"URL is not available or is not a JPEG image: {url}", file=err)
                        url_fails += 1


    pdq_verify_fails = 0
    hashes = df["pdq_hash"].tolist()
    """Iterate over the contents of a folder and compute PDQ hash for each file."""
    print("\nPDQ VERIFICATION:", file=err)
    for i, hash in enumerate(hashes):
        #image_format = urls[i].split('.')[-1]
        image_format = "jpg"
        file_path = os.path.join(folder_name, "%d.%s" % (i, image_format))
        if os.path.exists(file_path):
            pdq_hash = compute_pdqhash(file_path)
            if pdq_hash == "":
                pdq_verify_fails += 1
                if not args.keep_non_verified:
                    os.remove(file_path)
            else:
                dist = compute_pdqhash_distance(pdq_hash, hash)
                if dist > args.pdq_threshold:
                    print(f"PDQ VERIFICATION: File: {file_path}, PDQ mismatch: Hamming distance is {dist}, we remove the file", file=err)
                    pdq_verify_fails += 1
            
                    if not args.keep_non_verified:
                        os.remove(file_path)
        else:
            pdq_verify_fails += 1


    sha_verify_fails = 0
    hashes = df["sha256_hash"].tolist()
    """Iterate over the contents of a folder and compute SHA-256 for each file."""
    print("\nSHA256 VERIFICATION:", file=err)
    for i, hash in enumerate(hashes):
        if hash == "-":
            sha_verify_fails += 1
            continue
        #image_format = urls[i].split('.')[-1]
        image_format = "jpg"
        file_path = os.path.join(folder_name, "%d.%s" % (i, image_format))
        if os.path.exists(file_path):
            sha256_hash = calculate_sha256(file_path)
            if sha256_hash != hash:
                print(f"File: {file_path}, SHA-256 mismatch: {sha256_hash} != {hash}", file=err)
                sha_verify_fails += 1
        else:
            sha_verify_fails += 1

    print(f"Summary of data collection:")
    print(f"{len(urls) - url_fails}/{len(urls)} images downloaded successfully.")
    print(f"{len(urls) - pdq_verify_fails}/{len(urls) - url_fails} images verified successfully via PDQ. The non-verified images are removed.")
    print(f"{len(urls) - sha_verify_fails}/{len(urls) - pdq_verify_fails} images verified successfully via SHA-256 and match exactly. The non-verified images are kept.")
    print(f"Total number of images stored: {len(urls) - pdq_verify_fails}, {(((len(urls) - pdq_verify_fails) / len(df)) * 100):.2f}% of original images, errors are logged in errors.txt")


if __name__ == "__main__":
    main()

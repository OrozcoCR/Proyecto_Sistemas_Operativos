from os import path, makedirs, walk
from firebase_admin import credentials, initialize_app, storage

# Initialize
PWD_PATH = path.dirname(__file__)
CREDENTIALS = credentials.Certificate("./serviceAccount.json")
initialize_app(CREDENTIALS, {'storageBucket': "emotion-4c092.appspot.com"})



def create_folder_if_not_exists(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# Function to download everything for main bucket


def download_entire_bucket():
    bucket = storage.bucket()

    blobs = bucket.list_blobs()

    for blob in blobs:
        subfolder = path.dirname(blob.name)
        create_folder_if_not_exists(path.join(PWD_PATH, subfolder))
        local_path = path.join(PWD_PATH, blob.name)
        blob.download_to_filename(local_path)
        print(f'Downloaded: {blob.name}')

# Function to upload everything in a local folder to Firebase Storage


def upload_folder_contents():
    # Can change depending on data path
    directory_path = path.join(PWD_PATH, '..', "DATA")

    if not (path.exists(directory_path)):
        raise NotADirectoryError(f"Not directory '{directory_path}' to upload")

    bucket = storage.bucket()
    for root, _, files in walk(directory_path):
        for filename in files:
            local_path = path.join(root, filename)
            relative_path = path.relpath(local_path, PWD_PATH)
            blob = bucket.blob(relative_path)
            blob.upload_from_filename(local_path)
            print(f'Uploaded: {relative_path}')


# download_entire_bucket()
# upload_folder_contents()

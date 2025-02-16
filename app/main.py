# import requests

# response = requests.post(
#     "https://api.tryterra.co/v2/auth/generateWidgetSession",
#     headers={"x-api-key":"text","dev-id":"text","Content-Type":"application/json"},
#     json={"providers":"GARMIN,FITBIT,OURA,WITHINGS,SUUNTO","language":"en","reference_id":"user123@email.com","auth_success_redirect_url":"https://myapp.com/success","auth_failure_redirect_url":"https://myapp.com/failure"}
# )
# data = response.json()
# print(data)


import firebase_admin
from firebase_admin import credentials, firestore, storage
import os

# Replace 'path/to/your/serviceAccountKey.json' with the actual path
cred = credentials.Certificate("app/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Read data
users_ref = db.collection("users")  # Replace 'users' with your collection name
docs = users_ref.stream()
# for doc in docs:
# print(f'{doc.id} => {doc.to_dict()}')


# Write data
# new_user_ref = users_ref.add({
#     'first': 'Alan',
#     'last': 'Turing',
#     'born': 1912
# })
# print(f'Added user with ID: {new_user_ref[1].id}')

# Remember to shut down the app when you're done
firebase_admin.delete_app(firebase_admin.get_app())


# Initialize the Firebase Admin SDK.  Only do this once per application.
firebase_admin.initialize_app(
    cred,
    {
        "storageBucket": "hyd-ration-ut5jsu.firebasestorage.app"  # Replace with your bucket name
    },
)

# Get a reference to the Firebase Storage bucket.
bucket = storage.bucket()


def list_files(prefix=None):
    """Lists files in the bucket, optionally filtering by prefix."""
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        print(f"File: {blob.name}, Size: {blob.size} bytes")


def download_file(file_path, destination_path):
    """Downloads a file from Firebase Storage."""
    blob = bucket.blob(file_path)
    if blob.exists():
        blob.download_to_filename(destination_path)
        print(f"File '{file_path}' downloaded to '{destination_path}'")
    else:
        print(f"File '{file_path}' not found in Firebase Storage.")


def upload_file(file_path, destination_path):
    """Uploads a file to Firebase Storage."""
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file_path)
    print(f"File '{file_path}' uploaded to '{destination_path}'")


# Example Usage
list_files("")  # List files with "images" prefix
# download_file("images/myimage.jpg", "local/myimage.jpg")
# upload_file("local/new_image.png", "images/new_image.png")

# Remember to shut down the app when you're finished
firebase_admin.delete_app(firebase_admin.get_app())

# Create media directory if it doesn't exist
media_dir = os.path.join("app", "media")
os.makedirs(media_dir, exist_ok=True)


def download_all_files():
    """Downloads all files from Firebase Storage to media folder."""
    blobs = bucket.list_blobs()
    for blob in blobs:
        destination_path = os.path.join(media_dir, blob.name)
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)
        print(f"Downloaded: {blob.name}")


# Download all files
download_all_files()

# Cleanup
firebase_admin.delete_app(firebase_admin.get_app())

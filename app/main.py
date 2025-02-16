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
from datetime import datetime
import json

# Create directories if they don't exist
media_dir = os.path.join("app", "media")
os.makedirs(media_dir, exist_ok=True)

# Initialize Firebase Admin SDK once
cred = credentials.Certificate("app/serviceAccountKey.json")
firebase_admin.initialize_app(
    cred, {"storageBucket": "hyd-ration-ut5jsu.firebasestorage.app"}
)

# Get references
db = firestore.client()
bucket = storage.bucket()


def get_user_id_from_path(file_path):
    """Extract user ID from file path (users/{user_id}/uploads/...)"""
    parts = file_path.split("/")
    return parts[1] if len(parts) > 1 else None


def process_and_store_image_data():
    """Process images and store metadata in Firestore using batch operations"""
    blobs = bucket.list_blobs()
    results = {}
    batch = db.batch()

    for blob in blobs:
        destination_path = os.path.join(media_dir, blob.name)
        user_id = get_user_id_from_path(blob.name)

        if not user_id:
            continue

        # Get or create user's results document
        if user_id not in results:
            results[user_id] = {
                "images": [],
                "last_updated": datetime.now().isoformat(),
                "user_id": user_id,
            }

        # Process new images
        if not os.path.exists(destination_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            print(f"Downloaded new file: {blob.name}")

            image_data = {
                "path": blob.name,
                "created_at": (
                    blob.time_created.isoformat()
                    if blob.time_created
                    else datetime.now().isoformat()
                ),
                "size": blob.size,
                "content_type": blob.content_type,
            }
            results[user_id]["images"].append(image_data)

    # Store results in Firestore using batch
    for user_id, data in results.items():
        if data["images"]:
            doc_ref = db.collection("results").document(user_id)
            doc = doc_ref.get()

            if doc.exists:
                existing_data = doc.to_dict()
                all_images = existing_data.get("images", []) + data["images"]

                batch.update(
                    doc_ref,
                    {
                        "images": all_images,
                        "last_updated": data["last_updated"],
                        "total_images": len(all_images),
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    },
                )
            else:
                batch.set(
                    doc_ref,
                    {
                        "user_id": user_id,
                        "images": data["images"],
                        "last_updated": data["last_updated"],
                        "total_images": len(data["images"]),
                        "created_at": firestore.SERVER_TIMESTAMP,
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    },
                )

    # Commit the batch
    batch.commit()
    print("All documents updated successfully in batch")

    # Save results to output file
    output_path = "app/output_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


# Process images and store results
process_and_store_image_data()

# Cleanup
firebase_admin.delete_app(firebase_admin.get_app())

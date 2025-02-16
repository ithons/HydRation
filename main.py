# import requests

# response = requests.post(
#     "https://api.tryterra.co/v2/auth/generateWidgetSession",
#     headers={"x-api-key":"text","dev-id":"text","Content-Type":"application/json"},
#     json={"providers":"GARMIN,FITBIT,OURA,WITHINGS,SUUNTO","language":"en","reference_id":"user123@email.com","auth_success_redirect_url":"https://myapp.com/success","auth_failure_redirect_url":"https://myapp.com/failure"}
# )
# data = response.json()
# print(data)


import sys
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
import json
from google.auth.exceptions import RefreshError
from src.main import process_video
import time
import signal


def signal_handler(sig, frame):
    print("\nGracefully shutting down...")
    try:
        firebase_admin.delete_app(firebase_admin.get_app())
    except:
        pass
    sys.exit(0)


def main():
    try:
        # Create directories if they don't exist
        media_dir = os.path.join("media")
        os.makedirs(media_dir, exist_ok=True)

        # Verify the service account file exists
        service_account_path = "serviceAccountKey.json"
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                f"Service account file not found at {service_account_path}"
            )

        # Initialize Firebase Admin SDK once
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(
            cred, {"storageBucket": "hyd-ration-ut5jsu.firebasestorage.app"}
        )

        # Get references
        db = firestore.client()
        bucket = storage.bucket()

        print("Firebase initialized successfully")
        print("Monitoring for new videos... Press Ctrl+C to stop")

        # Track processed files to avoid reprocessing
        processed_files = set()

        while True:
            try:
                blobs = bucket.list_blobs()
                results = {}

                for blob in blobs:
                    # Skip if already processed
                    if blob.name in processed_files:
                        continue

                    destination_path = os.path.join(media_dir, blob.name)
                    user_id = (
                        blob.name.split("/")[1]
                        if len(blob.name.split("/")) > 1
                        else None
                    )

                    if not user_id:
                        processed_files.add(
                            blob.name
                        )  # Mark invalid files as processed
                        continue

                    # Initialize user results if not exists
                    if user_id not in results:
                        results[user_id] = {
                            "user_id": user_id,
                            "last_updated": datetime.now().isoformat(),
                        }

                    # Process new videos
                    if not os.path.exists(destination_path):
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        blob.download_to_filename(destination_path)
                        print(f"Downloaded new video: {blob.name}")

                        try:
                            video_results = process_video(destination_path)
                            # Store exact values from video processing
                            results[user_id].update(
                                {
                                    "heart_rate": (
                                        float(video_results[0])
                                        if hasattr(video_results[0], "item")
                                        else video_results[0]
                                    ),
                                    "hydration_score": (
                                        float(video_results[1])
                                        if hasattr(video_results[1], "item")
                                        else video_results[1]
                                    ),
                                    "hydration_class": video_results[2],
                                }
                            )
                            processed_files.add(blob.name)
                        except Exception as e:
                            print(f"Error processing video {blob.name}: {str(e)}")
                            print(f"Video results were: {video_results}")  # Debug info
                            continue

                # Update Firestore if there are results
                if results:
                    batch = db.batch()
                    for user_id, data in results.items():
                        doc_ref = db.collection("users").document(user_id)

                        try:
                            batch.set(
                                doc_ref,
                                {
                                    "heart_rate": data["heart_rate"],
                                    "hydration_score": data["hydration_score"],
                                    "hydration_class": data["hydration_class"],
                                    "last_updated": datetime.now().isoformat(),
                                    "updated_at": firestore.SERVER_TIMESTAMP,
                                },
                                merge=True,
                            )

                            batch.commit()
                            print(f"Updated measurements for user {user_id}")

                            # Save results locally
                            output_path = "output_results.json"
                            with open(output_path, "w") as f:
                                json.dump(results, f, indent=2)
                            print(f"Results saved to {output_path}")

                        except Exception as e:
                            print(
                                f"Error updating Firestore for user {user_id}: {str(e)}"
                            )

            except Exception as e:
                print(f"Error in main loop: {str(e)}")

            # Wait before next check
            time.sleep(1)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if firebase_admin._apps:
            firebase_admin.delete_app(firebase_admin.get_app())
        sys.exit(1)


if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    main()

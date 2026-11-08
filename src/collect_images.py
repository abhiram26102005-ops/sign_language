import cv2
import os
import time

# Signs/classes
SIGNS = [
    "please",
    "sorry",
    "help",
    "good",
    "clockit"
]

# Number of images to collect for each sign
IMAGES_PER_SIGN = 100

# Dataset folder
DATASET_PATH = "dataset"


def collect_images():
    # First, make sure dataset folders exist
    os.makedirs(DATASET_PATH, exist_ok=True)

    for sign in SIGNS:
        folder_path = os.path.join(DATASET_PATH, sign)

        # If a file exists with this name, stop and warn
        if os.path.isfile(folder_path):
            print(f"ERROR: {folder_path} is a file, not a folder.")
            print("Delete that file and create it as a folder.")
            return

        os.makedirs(folder_path, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    for sign in SIGNS:
        print(f"\nGet ready for sign: {sign.upper()}")
        print("Starting in 5 seconds...")
        time.sleep(5)

        count = 0

        while count < IMAGES_PER_SIGN:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to capture image.")
                break

            # Display text on frame
            cv2.putText(
                frame,
                f"Collecting: {sign} | {count + 1}/{IMAGES_PER_SIGN}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Image Collection", frame)

            # Save image
            image_name = f"{sign}_{count + 1}.jpg"
            image_path = os.path.join(DATASET_PATH, sign, image_name)
            cv2.imwrite(image_path, frame)

            count += 1

            # Small delay between images
            time.sleep(0.1)

            # Press q to stop early
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Stopped by user.")
                return

        print(f"Finished collecting images for: {sign.upper()}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nImage collection completed successfully.")


if __name__ == "__main__":
    collect_images()
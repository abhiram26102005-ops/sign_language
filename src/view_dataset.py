import cv2
import os
import random

DATASET_PATH = "dataset"


def view_random_images():
    signs = os.listdir(DATASET_PATH)

    for sign in signs:
        sign_path = os.path.join(DATASET_PATH, sign)

        if not os.path.isdir(sign_path):
            continue

        images = os.listdir(sign_path)

        if len(images) == 0:
            print(f"No images found for {sign}")
            continue

        random_image = random.choice(images)
        image_path = os.path.join(sign_path, random_image)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        cv2.putText(
            image,
            f"Class: {sign}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Dataset Viewer", image)
        print(f"Showing image: {image_path}")

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    view_random_images()
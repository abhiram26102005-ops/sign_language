import os
import random
import shutil

# Original folders
DATASET_PATH = "dataset"
ANNOTATIONS_PATH = "annotations"

# Output folders
TRAIN_IMAGES_PATH = "workspace/images/train"
TEST_IMAGES_PATH = "workspace/images/test"
TRAIN_ANNOTATIONS_PATH = "workspace/annotations/train"
TEST_ANNOTATIONS_PATH = "workspace/annotations/test"

# Train/test ratio
TRAIN_RATIO = 0.8

# Valid image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def create_output_folders():
    os.makedirs(TRAIN_IMAGES_PATH, exist_ok=True)
    os.makedirs(TEST_IMAGES_PATH, exist_ok=True)
    os.makedirs(TRAIN_ANNOTATIONS_PATH, exist_ok=True)
    os.makedirs(TEST_ANNOTATIONS_PATH, exist_ok=True)


def find_image_for_annotation(xml_file):
    base_name = os.path.splitext(xml_file)[0]

    for class_folder in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_folder)

        if not os.path.isdir(class_path):
            continue

        for ext in IMAGE_EXTENSIONS:
            image_path = os.path.join(class_path, base_name + ext)
            if os.path.exists(image_path):
                return image_path

    return None


def split_dataset():
    create_output_folders()

    xml_files = [
        file for file in os.listdir(ANNOTATIONS_PATH)
        if file.endswith(".xml")
    ]

    random.shuffle(xml_files)

    train_count = int(len(xml_files) * TRAIN_RATIO)

    train_files = xml_files[:train_count]
    test_files = xml_files[train_count:]

    print(f"Total annotations: {len(xml_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")

    for xml_file in train_files:
        image_path = find_image_for_annotation(xml_file)

        if image_path is None:
            print(f"WARNING: Image not found for {xml_file}")
            continue

        shutil.copy(image_path, os.path.join(TRAIN_IMAGES_PATH, os.path.basename(image_path)))
        shutil.copy(
            os.path.join(ANNOTATIONS_PATH, xml_file),
            os.path.join(TRAIN_ANNOTATIONS_PATH, xml_file)
        )

    for xml_file in test_files:
        image_path = find_image_for_annotation(xml_file)

        if image_path is None:
            print(f"WARNING: Image not found for {xml_file}")
            continue

        shutil.copy(image_path, os.path.join(TEST_IMAGES_PATH, os.path.basename(image_path)))
        shutil.copy(
            os.path.join(ANNOTATIONS_PATH, xml_file),
            os.path.join(TEST_ANNOTATIONS_PATH, xml_file)
        )

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    split_dataset()
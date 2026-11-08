import os
import glob
import csv
import xml.etree.ElementTree as ET


TRAIN_ANNOTATIONS_PATH = "workspace/annotations/train"
TEST_ANNOTATIONS_PATH = "workspace/annotations/test"

TRAIN_CSV_PATH = "workspace/annotations/train_labels.csv"
TEST_CSV_PATH = "workspace/annotations/test_labels.csv"


def xml_to_csv(annotations_path, output_csv_path):
    xml_files = glob.glob(os.path.join(annotations_path, "*.xml"))

    rows = []

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        for obj in root.findall("object"):
            class_name = obj.find("name").text

            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            rows.append([
                filename,
                width,
                height,
                class_name,
                xmin,
                ymin,
                xmax,
                ymax
            ])

    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ])

        writer.writerows(rows)

    print(f"Created CSV file: {output_csv_path}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    xml_to_csv(TRAIN_ANNOTATIONS_PATH, TRAIN_CSV_PATH)
    xml_to_csv(TEST_ANNOTATIONS_PATH, TEST_CSV_PATH)

    print("\nXML to CSV conversion completed successfully.")
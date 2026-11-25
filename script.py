from face_model import initialize_model, extract_features
from pathlib import Path
import numpy as np
import csv

model = initialize_model()


# used by save_dataset()
def parse_dataset(image_directory):

    root = Path(image_directory)

    for subject_directory in root.iterdir():

        # debug
        if not subject_directory.is_dir():
            continue

        id = subject_directory.name

        for image_path in sorted(subject_directory.glob("*.jpg")):

            # sample = imgX
            sample = image_path.stem.split("_")[-1]

            template = generate_template(image_path)

            # convert NP binary array to string for CSV
            bin_str = "".join(str(bit) for bit in template)

            yield id, sample, bin_str


# Parse image_directory, write data to csv_out.csv
def save_dataset(image_directory, csv_out):

    with open(csv_out, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["id", "sample", "template"])

        writer.writerows(parse_dataset(image_directory))

# each image gets a 512 bit binary string
def generate_template(image_path):

    embedding = extract_features(model, image_path)

    template = binarize(embedding)

    return template


# LT 0: 0, GTE 0: 1
def binarize(embedding):
    return (embedding >= 0).astype(np.uint8)

def main():

    # Parse galleryset and probeset, write corresponding binary templates to csv
    save_dataset("GallerySet", "GallerySet_binary.csv")
    save_dataset("ProbeSet", "ProbeSet_binary.csv")




if __name__ == '__main__':
    main()

















'''

powershell venv

python3 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
./.venv/Scripts/Activate.ps1

pip install -r requirements.txt




'''
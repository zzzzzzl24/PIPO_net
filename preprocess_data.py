import os
import time
import argparse
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--myops_path', type=str,
                   default='./samed_MyoPS', help='download path for MyoPS data')
parser.add_argument('--dst_path', type=str,
                   default='./samed_MyoPS', help='root dir for data')
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')
args = parser.parse_args()

# test_data = list(range(0, 80))
# hashmap = {1:1, 2:2, 3:0, 4:3}
# hashmap = {1:1, 2:0, 3:2, 4:0, 5:3}

def preprocess_train_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"./process/train_npz", exist_ok=True)

    a_min, a_max = 0, 255
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        name = image_file.split('/')[-1].replace(".nii.gz","")

        if str(name) in file_test:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        # counter = 1
        # for k in sorted(hashmap.keys()):
        #     assert counter == k
        #     counter += 1
        #     label_data[label_data == k] = hashmap[k]
        # image_data = np.transpose(image_data, (2, 1, 0))  # [D, W, H]
        # label_data = np.transpose(label_data, (2, 1, 0))

        for dep in range(D):
            save_path = f"./process/train_npz/{name}_slice{dep:03d}.npz"
            with open("train.txt", "a") as file:
                file.write(f"{name}_slice{dep:03d}\n")
            np.savez(save_path, label=label_data[:,:,dep], image=image_data[:,:,dep])
    pbar.close()


def preprocess_valid_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"./process/test_vol_h5", exist_ok=True)

    a_min, a_max = 0, 255

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        name = image_file.split('/')[-1].replace(".nii.gz","")

        if str(name) not in file_test:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        # H, W, D = image_data.shape

        # image_data = np.transpose(image_data, (2, 1, 0))
        # label_data = np.transpose(label_data, (2, 1, 0))

        # counter = 1
        # for k in sorted(hashmap.keys()):
        #     assert counter == k
        #     counter += 1
        #     label_data[label_data == k] = hashmap[k]

        save_path = f"./process/test_vol_h5/{name}.npy.h5"
        with open("test_vol.txt", "a") as file:
            file.write(f"{name}\n")
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()
    pbar.close()


if __name__ == "__main__":
    data_root = "./raw/t2w"
    filenames = []
    # String sort
    image_files = sorted(glob(f"{data_root}/image/*.nii.gz"))
    label_files = sorted(glob(f"{data_root}/label/*.nii.gz"))
    for filename in image_files:
        filename = filename.split('/')[-1].replace(".nii.gz","")
        with open("all.txt", "a") as file:
            file.write(f"{filename}.npy.h5\n")
        filenames.append(filename)
    file_test = filenames[0:95:5]
    preprocess_train_image(image_files, label_files)
    preprocess_valid_image(image_files, label_files)

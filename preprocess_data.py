import os
import argparse
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default='/data/fangdg/samed_Synapse', help='download path for Synapse data')
parser.add_argument('--dst_path', type=str,
                   default='/data/fangdg/samed_Synapse', help='root dir for data')
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')

# ===== fixed-stride 5-fold =====
parser.add_argument('--fold', type=int, default=0, help='0..4')
parser.add_argument('--num_folds', type=int, default=5, help='default 5 for fixed stride CV')
parser.add_argument('--run_all_folds', action='store_true', default=False)
args = parser.parse_args()


def preprocess_train_image(image_files, label_files, file_test, out_root):
    train_npz_dir = os.path.join(out_root, "train_npz")
    os.makedirs(train_npz_dir, exist_ok=True)

    a_min, a_max = 0, 255

    train_list_path = os.path.join(out_root, "train.txt")

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        name = os.path.basename(image_file).replace(".nii.gz", "")

        if name in file_test:
            continue

        image_data = nib.load(image_file).get_fdata().astype(np.float32)
        label_data = nib.load(label_file).get_fdata().astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            image_data = (image_data - a_min) / (a_max - a_min)

        _, _, D = image_data.shape

        for dep in range(D):
            case_slice = f"{name}_slice{dep:03d}"
            save_path = os.path.join(train_npz_dir, f"{case_slice}.npz")
            with open(train_list_path, "a") as f:
                f.write(f"{case_slice}\n")
            np.savez(save_path, label=label_data[:, :, dep], image=image_data[:, :, dep])
    pbar.close()


def preprocess_valid_image(image_files, label_files, file_test, out_root):
    test_h5_dir = os.path.join(out_root, "test_vol_h5")
    os.makedirs(test_h5_dir, exist_ok=True)

    a_min, a_max = 0, 255

    test_list_path = os.path.join(out_root, "test_vol.txt")

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        name = os.path.basename(image_file).replace(".nii.gz", "")

        if name not in file_test:
            continue

        image_data = nib.load(image_file).get_fdata().astype(np.float32)
        label_data = nib.load(label_file).get_fdata().astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            image_data = (image_data - a_min) / (a_max - a_min)

        save_path = os.path.join(test_h5_dir, f"{name}.npy.h5")
        with open(test_list_path, "a") as f:
            f.write(f"{name}\n")

        with h5py.File(save_path, 'w') as hf:
            hf['image'] = image_data
            hf['label'] = label_data
    pbar.close()


def run_one_fold(fold_id, image_files, label_files, filenames):
    assert args.num_folds == 5, "固定步长五折通常就是5折；如要改折数，需要改步长逻辑"
    assert 0 <= fold_id < 5

    # fixed stride split
    file_test = set(filenames[fold_id::5])

    # per-fold output
    out_root = f"/data/fangdg/P2/MyoPS++/process/fold_{fold_id}"
    os.makedirs(out_root, exist_ok=True)

    # clear list files to avoid append pollution
    open(os.path.join(out_root, "train.txt"), "w").close()
    open(os.path.join(out_root, "test_vol.txt"), "w").close()
    open(os.path.join(out_root, "all.txt"), "w").close()

    # write all.txt (optional)
    with open(os.path.join(out_root, "all.txt"), "a") as f:
        for fn in filenames:
            f.write(f"{fn}.npy.h5\n")

    preprocess_train_image(image_files, label_files, file_test, out_root)
    preprocess_valid_image(image_files, label_files, file_test, out_root)

    print(f"[Fold {fold_id}] Done. test={len(file_test)} train={len(filenames)-len(file_test)}")
    print(f"Outputs: {out_root}")


if __name__ == "__main__":
    data_root = "/data/fangdg/P2/MyoPS++/raw/t2w"
    image_files = sorted(glob(f"{data_root}/image/*.nii.gz"))
    label_files = sorted(glob(f"{data_root}/label/*.nii.gz"))
    assert len(image_files) == len(label_files), "image/label数量不一致"

    filenames = [os.path.basename(p).replace(".nii.gz", "") for p in image_files]

    if args.run_all_folds:
        for k in range(5):
            run_one_fold(k, image_files, label_files, filenames)
    else:
        run_one_fold(args.fold, image_files, label_files, filenames)

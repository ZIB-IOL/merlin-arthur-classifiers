import os
import random
from typing import Callable, Optional, Tuple

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sh
import torch
import torch as th
import torchvision
import wget
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, OneHotEncoder
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

# Comments are about number of NaN in training set
header_names = [
    "age",  # clean
    "workclass",  # 1836 NaN out of 32560 samples
    "fnlwgt",  # being dropped
    "education",  # clean
    "education-num",  # being dropped
    "marital-status",  # clean
    "occupation",  # 1843 NaN out of 32560 samples
    "relationship",  # clean
    "race",  # clean
    "sex",  # clean
    "capital-gain",  # clean
    "capital-loss",  # clean
    "hours-per-week",  # clean
    "native-country",  # 583 NaN out of 32560 samples
    "income",  # clean
]


class CustomMNIST(MNIST):
    """Modified MNIST dataset with additional masks for Merlin and Morgana."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Call super class constructor and add masks.

        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        super(CustomMNIST, self).__init__(root, train, transform, target_transform, download)  # ignore type error
        self.mask_merlin = (
            torch.empty_like(self.data, dtype=torch.float32).uniform_().unsqueeze(1)  # ignore type error
        )  # unsqueeze to get shape (N, 1, 28, 28)
        self.mask_morgana = (
            torch.empty_like(self.data, dtype=torch.float32).uniform_().unsqueeze(1)  # ignore type error
        )  # unsqueeze to get shape (N, 1, 28, 28)

    def get_mask(self, mode: str, idx: int) -> torch.Tensor:
        """Returns mask"""
        # Check input
        assert mode in ["merlin", "morgana"], f"mode needs to be merlin or morgana, got: {mode}"
        assert isinstance(idx, int), f"idx needs to be int, got: {type(idx)}"
        # Get mask
        if mode == "merlin":
            return self.mask_merlin[idx]
        elif mode == "morgana":
            return self.mask_morgana[idx]
        else:
            raise ValueError(f"mode needs to be merlin or morgana, got: {mode}")

    def set_mask(self, new_mask: torch.Tensor, mode: str, idx: torch.Tensor):
        """Sets mask"""
        # Check input
        assert mode in ["merlin", "morgana"], f"mode needs to be merlin or morgana, got: {mode}"
        assert isinstance(idx, torch.Tensor), f"idx needs to be int, got: {type(idx)}"
        assert isinstance(new_mask, torch.Tensor), f"new_mask needs to be torch.Tensor, got: {type(new_mask)}"
        assert new_mask.shape == self.data[idx].unsqueeze(1).shape, f"new_mask needs to have shape {self.data[idx].unsqueeze(1).shape}, got: {new_mask.shape}"  # fmt: skip
        # Set mask
        if mode == "merlin":
            self.mask_merlin[idx] = new_mask
        elif mode == "morgana":
            self.mask_morgana[idx] = new_mask
        else:
            raise ValueError(f"mode needs to be merlin or morgana, got: {mode}")

    def __getitem__(self, idx):
        """Returns image, label, merlin_mask, morgana_mask, idx.

        Args:
            idx (int): Index of image to return.

        Returns:
            image (torch.Tensor): Image.
            label (int): Label of image.
            merlin_mask (torch.Tensor): Merlin mask.
            morgana_mask (torch.Tensor): Morgana mask.
            idx (int): Index of image.
        """
        # Get image and label
        image, label = super(CustomMNIST, self).__getitem__(index=idx)
        # Get masks
        merlin_mask = self.get_mask(mode="merlin", idx=idx)
        morgana_mask = self.get_mask(mode="morgana", idx=idx)

        return image, label, merlin_mask, morgana_mask, idx


class CustomUCICensus(Dataset):
    def __init__(self, dataset, target_class, balance_dataset=True):
        """Construcs the Dataset given categorical US-Census-Income data.

        * Can also be used for sex classification task.

        Args:
            dataset (pandas.DataFrame): Census-Income Dataset.
            target_class (string): Target column name (e.g., "income" or "sex_target").

        Returns via __getitem__:
            label (torch.tensor): True target class
            x_input (torch.tensor): Census datapoint.
            mask (torch.tensor): Mask corresponding to datapoint.
            idx (int): Index corresponding to current items.
        """
        self.target_class = target_class
        self.labels = dataset[self.target_class].astype(np.int64).values
        self.dataset = dataset.drop(self.target_class, axis=1)

        if balance_dataset:
            under_sampler = RandomUnderSampler(random_state=42)
            self.dataset, self.labels = under_sampler.fit_resample(self.dataset, self.labels)  # type: ignore

        self.size = self.dataset.shape[0]

        # Random initialization of masks | th.zeros performs faster on testset
        n_features = len(self.dataset.iloc[0].values)  # type: ignore
        self.mask_merlin = th.zeros((self.size, n_features))
        self.mask_morgana = th.zeros((self.size, n_features))

    def set_mask(self, new_mask, mode, idx):
        """Replaces mask with optimized mask"""
        if mode == "merlin":
            self.mask_merlin[idx] = new_mask
        elif mode == "morgana":
            self.mask_morgana[idx] = new_mask
        else:
            raise ValueError(f"mode needs to be merlin or morgana, got: {mode}")

    def __len__(self):
        """Returns number of samples in dataset."""
        return self.size

    def __getitem__(self, idx):
        """Returns datapoints"""
        if self.target_class == "income":
            datapoint = self.dataset.iloc[idx].values  # type: ignore #
            self.x_input = datapoint[0]
            for row in datapoint[1:]:
                self.x_input = np.vstack((self.x_input, row))
            self.x_input = th.from_numpy(self.x_input).float()

        elif self.target_class == "sex_target":
            # target class is incorporated into datapoint
            datapoint = self.dataset.iloc[idx].values  # type: ignore #
            self.x_input = datapoint[0]
            for row in datapoint[1:]:
                self.x_input = np.vstack((self.x_input, row))
            self.x_input = th.from_numpy(self.x_input).float()

        return [self.x_input, self.labels[idx], self.mask_merlin[idx], self.mask_morgana[idx], idx]


class SVHNDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str,
        target_digit: Optional[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        balance: bool = False,
        max_samples: Optional[int] = None,
        one_vs_two: Optional[bool] = None,
        use_grayscale: Optional[bool] = None,
    ):
        """SVHN Dataset.

        Args:
            file_path (string): Path to the folder that contains the SVHN train, test or extra folders, e.g., ``.data/`.
            split (string): Split of the dataset. Needs to be "train" or "test".
            target_digit (int): Target digit to classify.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
            balance (bool, optional): Whether to balance the dataset or not.
            max_samples (int, optional): Maximum number of samples to use.
            one_vs_two (bool, optional): Whether to use the one vs two classification task or not.
            use_grayscale (bool, optional): Whether to use grayscale images or not.
        """
        self.split = split
        print("Loading SVHN data from", file_path, "for split", split, "...")
        print("Target digit:", target_digit)
        # Check if split is valid
        if self.split in ("train", "test", "extra"):
            file_path = os.path.join(file_path, f"{self.split}/digitStruct.mat")
        else:
            raise ValueError(f"split needs to be `train` or `test` or `extra`, got: {self.split}")

        # Load data in orignal format (hdf5)
        self.hdf5_data = h5py.File(file_path, "r")

        # Get target digit
        self.target_digit = target_digit
        self.transform = transform
        self.target_transform = target_transform
        self.balance = balance
        self.max_samples = max_samples
        self.one_vs_two = one_vs_two
        self.use_grayscale = use_grayscale

        assert self.target_digit in range(10), "target_digit needs to be in range(10)"

        # Get width and height of resizes from transforms
        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, torchvision.transforms.Resize):
                    self.resize_height, self.resize_width = t.size  # type: ignore
                    break
        else:
            raise ValueError("transform cannot be None")

        # Convert hdf5 file to pandas dataframe
        self.df = self._convert_to_dataframe()
        if self.balance:
            self._balance_dataset()

        assert self.resize_width is not None and self.resize_height is not None

        # Create empty masks for merlin and morgana
        self.mask_merlin = torch.empty(
            (len(self.df), 1, self.resize_height, self.resize_width), dtype=torch.float32
        ).uniform_()
        self.mask_morgana = torch.empty(
            (len(self.df), 1, self.resize_height, self.resize_width), dtype=torch.float32
        ).uniform_()

    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Converts the dataset to a pandas dataframe"""
        if self.split in ("train", "test", "extra"):
            print(
                f"Converting original {self.split} hdf5 file format to pandas dataframe... this might take a while ..."
            )
            print(
                "Alternatively, you can manually save the preprocessed dataframe to a csv file and load it from there."
            )
        # Get number of samples
        self.max_samples = self.max_samples if self.max_samples is not None else len(self.hdf5_data["/digitStruct/bbox"])  # type: ignore
        data = []
        for i in range(self.max_samples):
            img_name = self._get_name(i)
            bbox = self._get_bbox(i)
            data.append([img_name, bbox["label"], bbox["left"], bbox["top"], bbox["width"], bbox["height"]])

        # Create dataframe
        df = pd.DataFrame(data, columns=["img_name", "digits", "left", "top", "width", "height"])

        if self.one_vs_two is None:
            # Check if the target digit is in the label and add corresponding binary label for all samples
            df["binary_label"] = df["digits"].apply(lambda x: 1 if self.target_digit in x else 0)
        else:
            # Create y=1 if one is in x and two is not and y=0 if two is in x and one is not
            df["binary_label"] = df["digits"].apply(
                lambda x: 1
                if self.target_digit in x and 2 not in x
                else 0
                if 2 in x and self.target_digit not in x
                else 2
            )
            # Drop samples with no target digit
            df = df[df["binary_label"] != 2]

        return df

    def _balance_dataset(self) -> None:
        if self.split in ("train", "test", "extra"):
            print(f"Balancing {self.split} dataset by default...")
        # Balance the dataset with respect to binary label
        minority_class_count = min(self.df["binary_label"].value_counts())
        self.df = (
            self.df.groupby("binary_label")
            .apply(lambda x: x.sample(n=minority_class_count, random_state=42))
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.df)  # type: ignore

    def set_mask(self, new_mask, mode, idx) -> None:
        """Replaces mask with optimized mask"""
        if mode == "merlin":
            self.mask_merlin[idx] = new_mask
        elif mode == "morgana":
            self.mask_morgana[idx] = new_mask
        else:
            raise ValueError(f"mode needs to be merlin or morgana, got: {mode}")

    def __getitem__(self, index):
        # Get data from dataframe
        img_name = self.df.iloc[index]["img_name"]
        label = self.df.iloc[index]["binary_label"]
        bbox = {
            "digits": self.df.iloc[index]["digits"],
            "left": self.df.iloc[index]["left"],
            "top": self.df.iloc[index]["top"],
            "width": self.df.iloc[index]["width"],
            "height": self.df.iloc[index]["height"],
        }
        # Get image
        img = self._get_image(img_name)
        # Original image width and height
        W, H = img.size

        # Get the bounding box coordinates
        left = bbox["left"]
        top = bbox["top"]
        width = bbox["width"]
        height = bbox["height"]

        # Find the bounding box that contains all the bounding boxes and ensure that they are positive or zero
        x_left = max(0, min(left))
        y_top = max(0, min(top))
        x_right = min(left[-1] + width[-1], W)
        y_bottom = min(max(top) + max(height), H)

        # Choose a random cropping area that is within the original image and contains all the bounding boxes
        left_crop_coordinate = x_left - 10
        top_crop_coordinate = y_top - 10
        right_crop_coordinate = x_right + 10
        bottom_crop_coordinate = y_bottom + 10
        # 4-tuple defining the left, upper, right, and lower pixel
        cropping_area = (left_crop_coordinate, top_crop_coordinate, right_crop_coordinate, bottom_crop_coordinate)
        # Perform the cropping operation using the selected area
        img = img.crop(cropping_area)

        if self.transform:
            img = self.transform(img)

        if self.split == "test":
            # Update the bounding box coordinates to be relative to the cropped image
            left = [x - left_crop_coordinate for x in left]
            top = [y - top_crop_coordinate for y in top]

            # resized height and width due to transforms
            resized_height, resized_width = self.transform.transforms[0].size  # type: ignore

            # Update the bounding box coordinates to be scaled by the resize operation
            left = [int(round(x * resized_width / (right_crop_coordinate - left_crop_coordinate))) for x in left]
            top = [int(round(y * resized_height / (bottom_crop_coordinate - top_crop_coordinate))) for y in top]
            width = [int(round(w * resized_width / (right_crop_coordinate - left_crop_coordinate))) for w in width]
            height = [int(round(h * resized_height / (bottom_crop_coordinate - top_crop_coordinate))) for h in height]

            max_num_digit_ones = 3
            pad_value = -1

            # Filter the bounding box coordinates for the digit "1"
            one_indices = [i for i, digit in enumerate(bbox["digits"]) if digit == 1]
            left = [left[i] for i in one_indices]
            top = [top[i] for i in one_indices]
            width = [width[i] for i in one_indices]
            height = [height[i] for i in one_indices]

            # Pad the bounding box coordinate arrays with the pad_value
            one_bboxes_padded = list(zip(left, top, width, height)) + [(pad_value, pad_value, pad_value, pad_value)] * (
                max_num_digit_ones - len(left)
            )
            left, top, width, height = zip(*one_bboxes_padded)

            # plot the rectangles inside the image
            # fig, ax = plt.subplots(1)
            # ax.imshow(img.permute(1, 2, 0), vmin=img.min(), vmax=img.max())  # type: ignore
            # for i in range(len(left)):
            #     rect = patches.Rectangle(
            #         (left[i], top[i]), width[i], height[i], linewidth=1, edgecolor="r", facecolor="none"
            #     )
            #     ax.add_patch(rect)  # type: ignore
            # # save the plot
            # fig.savefig("plot.png")
            return (
                img,
                label,
                self.mask_merlin[index],
                self.mask_morgana[index],
                torch.tensor(left),
                torch.tensor(top),
                torch.tensor(width),
                torch.tensor(height),
                index,
            )
        else:
            return img, label, self.mask_merlin[index], self.mask_morgana[index], index

    def _get_name(self, index):
        """Get the image name for a given index, which is used to convert the dataset to a pandas dataframe"""
        name_ref = self.hdf5_data["/digitStruct/name"][index].item()  # type: ignore
        return "".join([chr(v[0]) for v in self.hdf5_data[name_ref]])  # type: ignore

    def _get_image(self, img_name):
        full_path = f".data/svhn_format1/{self.split}/{img_name}"
        if self.use_grayscale is True:
            img = Image.open(full_path).convert("L")
        else:
            img = Image.open(full_path).convert("RGB")
        return img

    def _get_bbox(self, index):
        """Get the bounding box coordinates for a given index, which is used to convert the dataset to a pandas dataframe"""
        attrs = {}
        item_ref = self.hdf5_data["/digitStruct/bbox"][index].item()  # type: ignore
        for key in ["label", "left", "top", "width", "height"]:
            attr = self.hdf5_data[item_ref][key]  # type: ignore
            values = (
                [self.hdf5_data[attr[i].item()][0][0].astype(int) for i in range(len(attr))]  # type: ignore
                if len(attr) > 1  # type: ignore
                else [attr[0][0]]  # type: ignore
            )
            attrs[key] = values
        return attrs


def read_census_data(
    PATH, target_class, read_pre_processed=True, remove_corr_feat=False, download=True, drop_sex_feat=False
):
    """Loads Dataset into Memory and removes useless columns.

    In addition, training and test datasets are adjusted to the required format.

    args:
    PATH (str): Path to data, which is not pre-processed.
    target_class (str): Target class name in Census DataFrame (e.g., "income" or "sex_target")
    read_pre_processed (bool, optional): If true, reads pre-processed data from local disk. Defaults to True.
    remove_corr_feat (bool, optional): Removes correlated features ("marital-status" nad "relationship").
    download (bool, optional): If true, the files are downloaded from the URL if they do not already exist.

    Returns:
        pd.DataFrame: Training and Test Dataset
    """

    if download == True:
        if os.path.isfile(".data/adult.data") and os.path.isfile(".data/adult.test"):
            print("Files already downloaded")
        else:
            if os.path.isdir(".data") == False:
                sh.mkdir(".data")  # type: ignore
            url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
            print("Downloading Train and Test Dataset")
            wget.download(url_train, out=".data")
            wget.download(url_test, out=".data")

    if read_pre_processed and os.path.isfile(f"{PATH[:-6]}/{target_class}_encoded_data_train.pkl"):
        # Read preprocessed data
        train_data = pd.read_pickle(f"{PATH[:-6]}/{target_class}_encoded_data_train.pkl")
        test_data = pd.read_pickle(f"{PATH[:-6]}/{target_class}_encoded_data_test.pkl")
    else:
        # Read dataframe and apply preprocessing (takes some seconds)
        df_train_data = pd.read_csv(f"{PATH}.data", sep=",", header=None, names=header_names, na_values=" ?")
        df_train_data = df_train_data.dropna()
        df_train_data = df_train_data.drop(columns=["fnlwgt", "education"], axis=1)

        # check following code with keep="first"
        # df_data_without_sex = df_train_data.drop(columns=["sex"])
        # df_data_without_sex.duplicated(keep=False).sum() # 5892
        # df_train_data.duplicated(keep=False).sum() # 5218

        df_test_data = pd.read_csv(f"{PATH}.test", sep=",", header=None, names=header_names, na_values=" ?")
        df_test_data = df_test_data.dropna()
        df_test_data = df_test_data.drop(columns=["fnlwgt", "education"], axis=1)

        # df_data_without_sex = df_test_data.drop(columns=["sex"])
        # df_data_without_sex.duplicated(keep=False).sum() # 1899
        # df_test_data.duplicated(keep=False).sum() # 1610

        # Remove "." after each entry of income in test set
        df_test_data["income"] = df_test_data["income"].astype(str).str.replace(r".", r"", regex=False)
        if target_class == "sex_target" and remove_corr_feat is True:
            # Remove features that are strong correlated with sex-feature
            df_train_data = df_train_data.drop(columns=["relationship", "marital-status"], axis=1)
            df_test_data = df_test_data.drop(columns=["relationship", "marital-status"], axis=1)

        train_data, test_data = preprocess_data(
            df_train_data,
            df_test_data,
            target_class,
            remove_corr_feat=remove_corr_feat,
            drop_sex_feat=drop_sex_feat,
        )

    return train_data, test_data


def preprocess_data(train_data, test_data, target_class, remove_corr_feat, drop_sex_feat, write_to_disk=True):
    """Preprocessing/Loading of the Datasets.

    * Applies customized one-hot-encodings. Fixed length of one-hot-encoded categorical features!

    * Each continuous feature is represented as a vector of same length like the one-hot-encodings,
      but each entry of the vector is set to the constant value of the continuous variable.

    * Leads to small performance drop wrt accuracy -- probably due to lengthening of the continuous variables,
      but can be compensated by more epochs.

    * Target class is either "income" or "sex".

    Args:
        train_data (pd.DataFrame): Train Dataset as DataFrame.
        test_data (pd.DataFrame): Test Dataset as DataFrame.
        target_class (str): Target class name in Census DataFrame (e.g., "income" or "sex_target").
        remove_corr_feat (bool): Decides whether correlated features are removed or not.
        write_to_disk (bool, optional): If true, saves the pre-processed dataframe to local dir. Defaults to True.

    Returns:
        train_data (pd.DataFrame): Preprocessed One-Hot-Encoded Train Dataset
        test_data (pd.DataFrame): Preprocessed One-Hot-Encoded Test Dataset
    """
    print("... Preprocess Data and store locally ...")
    if target_class == "sex_target":
        idx = 11 if remove_corr_feat is True else 13
        # Add another column serving as target column
        train_data.insert(idx, "sex_target", train_data["sex"], allow_duplicates=True)
        test_data.insert(idx, "sex_target", test_data["sex"], allow_duplicates=True)

        # Binary encoding for sex
        label_encoder = LabelBinarizer(neg_label=-1, pos_label=1)
        train_data["sex"] = label_encoder.fit_transform(train_data["sex"])
        test_data["sex"] = label_encoder.transform(test_data["sex"])

    if drop_sex_feat is True:
        # Drop sex column in train and test dataset
        train_data = train_data.drop(columns=["sex"], axis=1)
        test_data = test_data.drop(columns=["sex"], axis=1)

    # Split columns into categorical_columns / categorical_train_columns / continuous_columns
    categorical_columns = [cat for cat in train_data.columns if type(train_data[cat][0]) == str]

    # Separate categorical train features from target class
    categorical_train_columns = [cat for cat in categorical_columns if cat != target_class]
    continuous_columns = [col for col in train_data.columns if col not in categorical_columns]

    scale_columns = continuous_columns[:]
    if target_class == "sex_target" and not drop_sex_feat:
        scale_columns.remove("sex")

    scaler = MinMaxScaler()
    train_data[scale_columns] = scaler.fit_transform(train_data[scale_columns])
    test_data[scale_columns] = scaler.transform(test_data[scale_columns])

    # Apply one-hot-encodings to categorical features and replace columns with encodings
    one_hot_encoders = {}
    for cat_col in categorical_train_columns:
        one_hot_encoders[cat_col] = OneHotEncoder(sparse=False)  # type: ignore
        traincol_encodings = one_hot_encoders[cat_col].fit_transform(train_data[cat_col].values.reshape(-1, 1)).tolist()
        train_data[cat_col] = traincol_encodings
        testcol_encodings = one_hot_encoders[cat_col].transform(test_data[cat_col].values.reshape(-1, 1)).tolist()
        test_data[cat_col] = testcol_encodings

    # Encode target classes
    label_encoder = LabelBinarizer()
    train_data[target_class] = label_encoder.fit_transform(train_data[target_class])
    test_data[target_class] = label_encoder.transform(test_data[target_class])

    # "native-country" is class with most categories
    max_encoding_size = len(one_hot_encoders["native-country"].categories_[0])

    train_data = convert_dataset_to_correct_format(
        train_data, continuous_columns, categorical_train_columns, max_encoding_size
    )
    test_data = convert_dataset_to_correct_format(
        test_data, continuous_columns, categorical_train_columns, max_encoding_size
    )

    if write_to_disk:
        train_data.to_pickle(path=f".data/{target_class}_encoded_data_train.pkl")
        test_data.to_pickle(path=f".data/{target_class}_encoded_data_test.pkl")

    return train_data, test_data


def convert_dataset_to_correct_format(dataset, continuous_columns, categorical_train_columns, max_encoding_size):
    """Converts UCI Census Dataset to the required format for ArthurMerlin framework.

    Resizes continuous variables to vector of fixed length, where each entry is the continuous variable.
    Pads one-hot-encoded categorical data with zeros to same fixed length. This leads to
    repetition of information, making it more difficult for the model to filter out
    patterns from the data.

    Args:
        dataset (pd.DataFrame): UCI Census Data
        continuous_columns (list): List of str containing the headers of continuous features.
        categorical_train_columns (list): List of str containing the headers of categorical features.
        max_encoding_size (int): Max length of one-hot-encoded features along all features.

    Returns;
        dataset (pd.DataFrame): Converted dataset.
    """

    for category in continuous_columns:
        dataset[category] = dataset.apply(lambda row: np.resize(row[category], new_shape=max_encoding_size), axis=1)
    for category in categorical_train_columns:
        dataset[category] = dataset.apply(lambda row: np.array(row[category]), axis=1)

    for category in categorical_train_columns:
        dataset[category] = dataset.apply(
            lambda row: np.pad(
                row[category],
                pad_width=(0, max_encoding_size - row[category].shape[0]),
                constant_values=0,
            ),
            axis=1,
        )
    return dataset

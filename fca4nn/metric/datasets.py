import os
import pandas as pd
from glob import glob
import pickle
from processing.imagenet100_classes import IMAGENET100_CLASSES
from torch.utils.data import Dataset

from PIL import Image
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop,
)


INET100_DATA_DIR = "/raid/DATASETS/inet100"
CUB200_DATA_DIR = "/raid/DATASETS/CUB_200_2011"
AWA2_DATA_DIR = "/raid/DATASETS/awa2"
CIFAR100_DATA_DIR = "/raid/DATASETS/cifar100"


class Cifar100Dataset(Dataset):
    def __init__(self, data_root=CIFAR100_DATA_DIR, split="train"):
        super().__init__()
        self.num_classes = 100
        self.split = split
        if split == "train":
            self.data = CIFAR100(
                root=data_root,
                train=True,
                download=True,
                transform=Compose(
                    [
                        RandomResizedCrop(224, interpolation=Image.BILINEAR),
                        RandomHorizontalFlip(),
                        ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                        ),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                ),
            )
        else:
            self.data = CIFAR100(
                root=data_root,
                train=False,
                download=True,
                transform=Compose(
                    [
                        Resize(size=256, interpolation=Image.BILINEAR),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                ),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label


class Inet100Dataset(Dataset):
    def __init__(
        self,
        data_root=INET100_DATA_DIR,
        split="train",
    ):
        self.num_classes = len(IMAGENET100_CLASSES.keys())
        self.split = split
        self.dir_idx = {k: v for v, k in enumerate(IMAGENET100_CLASSES.keys())}

        if split == "train":
            self.transforms = Compose(
                [
                    RandomResizedCrop(224, interpolation=Image.BILINEAR),
                    RandomHorizontalFlip(),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transforms = Compose(
                [
                    Resize(size=256, interpolation=Image.BILINEAR),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.data = []
        for d in IMAGENET100_CLASSES.keys():
            label = self.dir_idx[d]
            if split == "train":
                images = glob(os.path.join(data_root, "train", d, "*.JPEG"))
            elif split == "val":
                images = glob(os.path.join(data_root, "val", d, "*.JPEG"))
            elif split == "test":
                images = glob(os.path.join(data_root, "test_set", d, "*.JPEG"))
            else:
                raise ValueError("Invalid split: {}".format(split))
            self.data.extend(list(zip(images, [label] * len(images))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = Image.open(image).convert("RGB")
        image = self.transforms(image)
        return image, label


class CUB112Dataset(Dataset):
    def __init__(self, data_root=CUB200_DATA_DIR, split="train"):
        self.data_dir = data_root
        self.split = split
        self.num_classes = 200
        data_path = os.path.join(self.data_dir, "CUB112", f"{split}.pkl")
        if os.path.exists(data_path):
            self.data = pickle.load(open(data_path, "rb"))

        if split == "train":
            self.transform = Compose(
                [
                    ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                    RandomResizedCrop(224),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                ]
            )
        else:
            self.transform = Compose(
                [
                    Resize(size=256, interpolation=Image.BILINEAR),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]

        try:
            img_path = os.path.join(self.data_dir, img_path.split("/CUB_200_2011/")[-1])
            img = Image.open(img_path).convert("RGB")
        except:
            img_path_split = img_path.split("/")
            split = "train" if self.is_train else "test"
            img_path = "/".join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert("RGB")

        class_label = img_data["class_label"]
        if self.transform:
            img = self.transform(img)

        return img, class_label


class AwA2Dataset(Dataset):
    def __init__(
        self,
        data_root=AWA2_DATA_DIR,
        split="train",
    ):
        self.data_dir = data_root
        self.split = split
        self.num_classes = 50

        if split == "train":
            self.transform = Compose(
                [
                    RandomResizedCrop(224, interpolation=Image.BILINEAR),
                    RandomHorizontalFlip(),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = Compose(
                [
                    Resize(256, interpolation=Image.BILINEAR),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        df = pd.read_csv(os.path.join(self.data_dir, "{}.csv".format(split)))
        self.img_names = df["path"].tolist()
        self.img_index = df["label"].tolist()

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index].split("//")[-1])
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)

        label = self.img_index[index]
        return im, label

    def __len__(self):
        return len(self.img_names)

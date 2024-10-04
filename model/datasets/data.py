from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize
from model.datasets.uie_dataset import DatasetFromFolderEval,  DatasetFromFolder, transform, DatasetFromFolderTest


def get_training_set(dataset_path, data, label, patch_size, data_augmentation, image_size=256):
    data_dir = join(dataset_path, data)
    label_dir = join(dataset_path, label)

    return DatasetFromFolder(label_dir, data_dir, patch_size, data_augmentation, transform=transform(), image_size=image_size)


def get_eval_set(dataset_path, data, label, image_size):
    data_dir = join(dataset_path, data)
    label_dir = join(dataset_path, label)

    return DatasetFromFolderEval(data_dir, label_dir, transform=transform(), image_size=image_size)

def get_test_set(dataset_path, data):
    data_dir = join(dataset_path, data)

    return DatasetFromFolderTest(data_dir, transform=transform())


from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torch.nn.functional as F
import os
import numpy as np
from copy import deepcopy

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])


def lambda_transforms(x):
    return F.pad(x.unsqueeze(0), [4, 4, 4, 4], mode='reflect').squeeze()


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda_transforms),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

"""## Prepare data (download)"""


def get_CIFAR_data(cifar=10, train_transform=None, test_transform=None,
                   train_target_transform=None, test_target_transform=None):
    if cifar == 10:
        dataset = CIFAR10
        path = os.path.join("Data", "cifar-10-batches-py")
    elif cifar == 100:
        dataset = CIFAR100
        path = os.path.join("Data", "cifar-100-python")

    print("Training Data")
    train_data = dataset(path, train=True,
                         transform=train_transform,
                         target_transform=train_target_transform,
                         download=True)

    print("Testing Data")
    test_data = dataset(path, train=False,
                        transform=test_transform,
                        target_transform=test_target_transform,
                        download=True)

    return train_data, test_data


"""## Generate Imbalance Data"""


def generate_imbalance_data(dataset,
                            factor=200,
                            num_meta_per_class=10,
                            num_of_corrupted=10,
                            seed=123):
    imbalance_factor = int(factor)

    num_of_classes = np.unique(dataset.targets).shape[0]

    indices_dict = {target: np.where(np.array(dataset.targets) == target)[0]
                    for target in range(num_of_classes)}

    meta_indices = []
    train_indices = []
    np.random.seed(seed)
    for target in indices_dict.keys():
        np.random.shuffle(indices_dict[target])
        meta_indices += indices_dict[target][: num_meta_per_class].tolist()

        imbalance_size = int((indices_dict[target].shape[0] - num_meta_per_class) / float(imbalance_factor) ** float(
            target / (num_of_classes - 1)))

        train_indices += indices_dict[target][num_meta_per_class: num_meta_per_class + imbalance_size].tolist()

    train_data = deepcopy(dataset)
    meta_data = deepcopy(dataset)

    train_indices = np.array(train_indices).flatten()
    meta_indices = np.array(meta_indices).flatten()

    train_data.data = train_data.data[train_indices]
    meta_data.data = meta_data.data[meta_indices]

    train_data.targets = np.array(train_data.targets)[train_indices]
    meta_data.targets = np.array(meta_data.targets)[meta_indices]

    return train_data, meta_data, None


"""## Generate Uniform Noise Data"""


def generate_noise_data(dataset,
                        factor=0.4,
                        num_meta_per_class=10,
                        num_of_corrupted=10,
                        seed=123):
    num_of_classes = np.unique(dataset.targets).shape[0]

    indices_array = np.array([np.where(np.array(dataset.targets) == target)[0]
                              for target in range(num_of_classes)])

    temp_indices = np.empty((num_of_classes, indices_array.shape[1] - num_meta_per_class))
    meta_indices = []
    np.random.seed(seed)
    for n in range(num_of_classes):
        np.random.shuffle(indices_array[n])
        meta_indices.append(indices_array[n, : num_meta_per_class])
        temp_indices[n] = indices_array[n, num_meta_per_class:]

    train_data = deepcopy(dataset)
    meta_data = deepcopy(dataset)

    train_indices = temp_indices.flatten().astype(int)
    np.random.shuffle(train_indices)
    meta_indices = np.array(meta_indices).flatten().astype(int)

    train_data.data = train_data.data[train_indices]
    meta_data.data = meta_data.data[meta_indices]

    train_data.targets = np.array(train_data.targets)[train_indices]
    meta_data.targets = np.array(meta_data.targets)[meta_indices]

    corrupted_indices = []
    chosen_indices = [[x for x in range(num_of_classes) if x != t] for t in range(num_of_classes)]
    for i in range(num_of_classes):
        target_indices = np.where(train_data.targets == i)[0]
        for n in target_indices:
            if np.random.random() < factor:
                train_data.targets[n] = np.random.choice(chosen_indices[i], 1)
                corrupted_indices.append(n)

    np.random.shuffle(corrupted_indices)
    mask = np.full(train_data.targets.shape[0], False)
    mask[corrupted_indices[:num_of_corrupted]] = True

    corrupted_data = deepcopy(train_data)

    corrupted_data.data = deepcopy(train_data.data[mask])
    corrupted_data.targets = deepcopy(train_data.targets[mask])

    return train_data, meta_data, corrupted_data


"""## Generate Flip Noise Data"""


def generate_flip_data(dataset,
                       factor=0.4,
                       num_meta_per_class=10,
                       num_of_corrupted=10,
                       seed=123):
    num_of_classes = np.unique(dataset.targets).shape[0]

    indices_array = np.array([np.where(np.array(dataset.targets) == target)[0]
                              for target in range(num_of_classes)])

    temp_indices = np.empty((num_of_classes, indices_array.shape[1] - num_meta_per_class))
    meta_indices = []
    np.random.seed(seed)
    for n in range(num_of_classes):
        np.random.shuffle(indices_array[n])
        meta_indices.append(indices_array[n, : num_meta_per_class])
        temp_indices[n] = indices_array[n, num_meta_per_class:]

    train_data = deepcopy(dataset)
    meta_data = deepcopy(dataset)

    train_indices = temp_indices.flatten().astype(int)
    np.random.shuffle(train_indices)
    meta_indices = np.array(meta_indices).flatten().astype(int)

    train_data.data = train_data.data[train_indices]
    meta_data.data = meta_data.data[meta_indices]

    train_data.targets = np.array(train_data.targets)[train_indices]
    meta_data.targets = np.array(meta_data.targets)[meta_indices]

    classes = list(range(num_of_classes))
    np.random.shuffle(classes)

    random_pair_classes = [[i, j] for i, j in zip(
        classes[: int(len(classes) / 2)],
        classes[int(len(classes) / 2):])]

    corrupted_indices = []
    for pair in random_pair_classes:
        target_indices_0 = np.where(train_data.targets == pair[0])[0]
        np.random.shuffle(target_indices_0)
        target_indices_1 = np.where(train_data.targets == pair[1])[0]
        np.random.shuffle(target_indices_1)

        size = int(target_indices_0.shape[0] * factor)

        train_data.targets[target_indices_0[: size]] = deepcopy(pair[1])
        train_data.targets[target_indices_1[: size]] = deepcopy(pair[0])

        corrupted_indices += target_indices_0[: size].tolist() + target_indices_1[: size].tolist()

    np.random.shuffle(corrupted_indices)
    mask = np.full(train_data.targets.shape[0], False)
    mask[corrupted_indices[:num_of_corrupted]] = True

    corrupted_data = deepcopy(train_data)

    corrupted_data.data = deepcopy(train_data.data[mask])
    corrupted_data.targets = deepcopy(train_data.targets[mask])

    return train_data, meta_data, corrupted_data


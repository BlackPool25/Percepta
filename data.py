import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Standard MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

TASK_CLASSES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def _get_filtered_loader(
    dataset: datasets.MNIST,
    allowed_classes: tuple,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    indices = [
        i for i, label in enumerate(dataset.targets)
        if label in allowed_classes
    ]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_split_mnist_tasks(
    batch_size: int = 128,
    data_root: str = "./data",
) -> list[tuple[DataLoader, DataLoader]]:
    train_full = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_full = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    tasks = []
    for class_a, class_b in TASK_CLASSES:
        train_loader = _get_filtered_loader(
            train_full, (class_a, class_b), batch_size, shuffle=True
        )
        test_loader = _get_filtered_loader(
            test_full, (class_a, class_b), batch_size, shuffle=False
        )
        tasks.append((train_loader, test_loader))
    return tasks

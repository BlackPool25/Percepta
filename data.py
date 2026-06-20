import torch
import torch.nn.functional as F
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

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


class PermutePixels:
    """Pixel permutation transform for Permuted MNIST."""
    def __init__(self, permutation: torch.Tensor):
        self.perm = permutation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)[self.perm].view(1, 28, 28)


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
        root=data_root, train=True, download=True, transform=_base_transform
    )
    test_full = datasets.MNIST(
        root=data_root, train=False, download=True, transform=_base_transform
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


def get_permuted_mnist_tasks(
    n_tasks: int = 10,
    batch_size: int = 128,
    data_root: str = "./data",
    seed: int = 42,
) -> list[tuple[DataLoader, DataLoader]]:
    """Each task applies a different fixed random permutation to pixels.

    Labels stay 0-9 (domain-incremental, not class-incremental).
    Tests the model's ability to adapt to changing input distributions
    while maintaining digit classification performance.

    This is the core benchmark for the fast/slow architecture because:
    - Fast layer's value is within-task adaptation speed (new permutation)
    - Slow layer's value is cross-task retention (digit invariance)
    """
    rng = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(28 * 28, generator=rng) for _ in range(n_tasks)]

    tasks = []
    for perm in perms:
        train_transform = transforms.Compose([
            _base_transform,
            PermutePixels(perm),
        ])
        test_transform = transforms.Compose([
            _base_transform,
            PermutePixels(perm),
        ])
        train_set = datasets.MNIST(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = datasets.MNIST(
            root=data_root, train=False, download=True, transform=test_transform
        )
        tasks.append((
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        ))
    return tasks, perms


def generate_drift_stream(
    n_frames: int = 10000,
    batch_size: int = 8,
    blur_range: tuple[float, float] = (0.0, 4.0),
    seed: int = 42,
) -> list[DataLoader]:
    """Generate a continuous single-pass stream with gradual visual drift.

    Each frame is an MNIST digit with Gaussian blur that oscillates
    smoothly between 0 and max_sigma. Consecutive frames have almost
    identical blur — this temporal continuity is what episodic retrieval
    needs to exploit.

    Returns list of DataLoaders (one per 'task' to match existing API),
    but each yields frames sequentially from the drift stream.
    """
    rng = torch.Generator().manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)

    stream_x, stream_y = [], []
    blur_sigma = 0.0
    direction = 0.05

    for i in range(n_frames):
        idx = torch.randint(len(mnist), (1,), generator=rng).item()
        img, label = mnist[idx]

        if blur_sigma > 0.01:
            kernel_size = max(3, int(2 * blur_sigma) | 1)
            img = transforms.functional.gaussian_blur(
                img.unsqueeze(0), kernel_size=kernel_size, sigma=[blur_sigma, blur_sigma]
            ).squeeze(0)

        stream_x.append(img)
        stream_y.append(label)

        blur_sigma += direction
        if blur_sigma >= blur_range[1] or blur_sigma <= blur_range[0]:
            direction = -direction

    all_x = torch.stack(stream_x)
    all_y = torch.tensor(stream_y)

    ds = torch.utils.data.TensorDataset(all_x, all_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return [(dl, dl)]

import os
import torch
import torch.nn.functional as F
import PIL.Image
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


# ─── CIFAR-10 Kaggle ──────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

SPLIT_CIFAR10_TASKS = [
    (0, 1),  # airplane, automobile
    (2, 3),  # bird, cat
    (4, 5),  # deer, dog
    (6, 7),  # frog, horse
    (8, 9),  # ship, truck
]

_CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR_STD = [0.2470, 0.2435, 0.2616]

_cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
])


def _load_kaggle_cifar10(root: str, force_reload: bool = False):
    """Load or load cached CIFAR-10 tensors. Returns (data, labels)."""
    cache_file = os.path.join(root, 'cifar10_cache.pt')
    if not force_reload and os.path.exists(cache_file):
        cached = torch.load(cache_file)
        print(f'  Loaded CIFAR-10 from cache ({len(cached[0])} samples)')
        return cached

    img_dir = os.path.join(root, 'cifar10_train', 'train')
    label_file = os.path.join(root, 'trainLabels.csv')
    with open(label_file) as f:
        lines = f.read().strip().split('\n')[1:]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])

    tensors, labels = [], []
    for line in lines:
        idx, label_name = line.split(',')
        img = PIL.Image.open(os.path.join(img_dir, f'{idx}.png')).convert('RGB')
        tensors.append(transform(img))
        labels.append(CIFAR10_CLASSES.index(label_name))

    data = (torch.stack(tensors), torch.tensor(labels))
    torch.save(data, cache_file)
    print(f'  Cached CIFAR-10 ({len(data[0])} samples)')
    return data


class KaggleCIFAR10(torch.utils.data.Dataset):
    """Fast cached CIFAR-10 dataset (data is pre-normalized)."""
    def __init__(self, root: str = './data', train: bool = True,
                 force_reload: bool = False):
        self.data, self.labels = _load_kaggle_cifar10(root, force_reload)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_split_cifar10_tasks(
    batch_size: int = 128,
    data_root: str = './data',
) -> list[tuple[DataLoader, DataLoader]]:
    """Split-CIFAR-10: 5 tasks, 2 classes each (class-incremental)."""
    full = KaggleCIFAR10(root=data_root, train=True)
    test_full = KaggleCIFAR10(root=data_root, train=True)

    tasks = []
    for c1, c2 in SPLIT_CIFAR10_TASKS:
        train_idx = [i for i, l in enumerate(full.labels) if l in (c1, c2)]
        test_idx = [i for i, l in enumerate(test_full.labels) if l in (c1, c2)]
        train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(test_full, test_idx), batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))
    return tasks


def generate_cifar_drift_stream(
    n_frames: int = 10000,
    batch_size: int = 8,
    data_root: str = './data',
    seed: int = 42,
) -> list[tuple[DataLoader, DataLoader]]:
    """Continuous single-pass stream with gradual visual drift on CIFAR-10.

    Applies smoothly oscillating blur, color jitter, and rotation.
    Temporal continuity enables episodic retrieval and logit distillation.
    """
    rng = torch.Generator().manual_seed(seed)
    base = KaggleCIFAR10(root=data_root, train=True)

    stream_x, stream_y = [], []
    blur_sigma, bright, contrast, rot = 0.0, 0.0, 0.0, 0.0
    speed = 0.1

    for i in range(n_frames):
        idx = torch.randint(len(base), (1,), generator=rng).item()
        img, label = base[idx]

        # Apply continuous transforms
        if blur_sigma > 0.02:
            ks = max(3, int(2 * blur_sigma) | 1)
            img = transforms.functional.gaussian_blur(
                img.unsqueeze(0), ks, [blur_sigma, blur_sigma]).squeeze(0)
        img = transforms.functional.adjust_brightness(img, 1.0 + bright)
        img = transforms.functional.adjust_contrast(img, 1.0 + contrast)
        img = transforms.functional.rotate(img.unsqueeze(0), rot).squeeze(0)

        stream_x.append(img)
        stream_y.append(label)

        # Oscillate drift parameters
        blur_sigma += speed * 0.3
        bright += speed * 0.01
        contrast += speed * 0.01
        rot += speed * 2.0

        for attr in ['blur_sigma', 'bright', 'contrast', 'rot']:
            val = locals()[attr]
            var = {'blur_sigma': 3.0, 'bright': 0.3, 'contrast': 0.3, 'rot': 30.0}[attr]
            if abs(val) > var:
                speed = -speed
                break

    all_x = torch.stack(stream_x)
    all_y = torch.tensor(stream_y)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return [(dl, dl)]

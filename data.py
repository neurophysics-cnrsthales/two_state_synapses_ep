import torch
import torchvision


class GPUBatches:
    """Iterates over tensors that already live on the compute device (no CPU DataLoader overhead)."""

    def __init__(self, x, y, batch_size, shuffle):
        self.x, self.y, self.batch_size, self.shuffle = x, y, batch_size, shuffle

    def __len__(self):
        return (self.x.size(0) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.x.size(0)
        order = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield self.x[idx], self.y[idx]


def mnist_gpu_loaders(root, batch_size, batch_size_test, n_output, device, val_split=0):
    """MNIST as flat float tensors on `device`. Targets: one-hot (repeated n_output//10 times) for train, ints for eval.

    val_split > 0 holds that many training images out as a validation set (fixed split, independent of the
    training seed) and returns it in place of the test set, so hyper-parameter tuning never touches the test set.
    """
    def load(train):
        ds = torchvision.datasets.MNIST(root=root, train=train, download=True)
        return ds.data.float().div(255).reshape(len(ds), -1).to(device), ds.targets.to(device)

    x_tr, y_tr = load(True)
    if val_split > 0:
        order = torch.randperm(x_tr.size(0), generator=torch.Generator().manual_seed(12345)).to(device)
        val_idx, tr_idx = order[:val_split], order[val_split:]
        x_te, y_te = x_tr[val_idx], y_tr[val_idx]
        x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
    else:
        x_te, y_te = load(False)
    y_tr = torch.nn.functional.one_hot(y_tr, 10).float().repeat_interleave(n_output // 10, dim=1)
    return GPUBatches(x_tr, y_tr, batch_size, True), GPUBatches(x_te, y_te, batch_size_test, False)

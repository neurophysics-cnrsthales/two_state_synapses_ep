import torch
import torch.nn as nn


def random_sign(device):
    """+1 / -1 with equal probability. torch.sign(randn) returns 0 when randn is exactly 0, which would set
    beta to 0 for the rest of training (abs(0) = 0) and turn every weight update into NaN."""
    return (torch.randn(1, device=device) >= 0).float() * 2 - 1


def train_dyn_ep(net, train_loader, optimizer):
    """Phased update: free phase, nudge phase, then ONE weight update per mini-batch."""
    correct_train = 0
    total_train = 0
    criterion = nn.MSELoss()
    current_loss = 0.0
    batch_id = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        s = net.init_state(data)
        net.beta = random_sign(net.device) * abs(net.beta)       # random sign of the nudge for every batch

        net.init_w()
        s, w_free = net(s, target=None)                          # free phase
        correct_train += (torch.argmax(s[0], dim=1) == torch.argmax(target, dim=1)).sum().float()
        total_train += target.size(dim=0)
        with torch.no_grad():
            current_loss += criterion(s[0], target).detach()     # MSE of the free-phase output (monitoring only)
            batch_id += 1

        s, w_nudge = net(s, target=target)                       # nudge phase
        net.update_W(w_free, w_nudge)
        optimizer.step()

    return 1 - correct_train / total_train, float(current_loss / batch_id)


def train_dyn_sequential_ep(net, train_loader, optimizer):
    """Continuous update: the weights are updated after EVERY step of the nudge phase."""
    correct_train = 0
    total_train = 0
    criterion = nn.MSELoss()
    current_loss = 0.0
    batch_id = 0

    for data, target in train_loader:
        s = net.init_state(data)
        net.beta = random_sign(net.device) * abs(net.beta)

        net.init_w()
        s, w_prev = net(s, target=None)                          # free phase
        correct_train += (torch.argmax(s[0], dim=1) == torch.argmax(target, dim=1)).sum().float()
        total_train += target.size(dim=0)
        with torch.no_grad():
            current_loss += criterion(s[0], target).detach()
            batch_id += 1

        for t in range(net.Kmax):                                # nudge phase, one update per step
            optimizer.zero_grad()
            with torch.no_grad():
                s = net.update_s(s, target=target)
                net.dyn = net.instarD(net.dyn, s)
            net.update_W(w_prev, net.dyn)
            optimizer.step()
            w_prev = [item.clone() for item in net.dyn]

    return 1 - correct_train / total_train, float(current_loss / batch_id)


def test_dyn_ep(net, test_loader):
    """Test (or validation) error of the free-phase output."""
    correct_test = 0
    total_test = 0

    for data, target in test_loader:
        s = net.init_state(data)
        net.init_w()
        s, _ = net(s, target=None)
        correct_test += (torch.argmax(s[0], dim=1) == target).sum().float()
        total_test += target.size(dim=0)

    return 1 - correct_test / total_test

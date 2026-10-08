import torch
import torch.nn as nn


class InstarD(nn.Module):
    """Dynamic weights w (one matrix per layer pair): w <- clip(w + dt * (V - w), 0, 1),
    with V the batch-averaged product of the pre- and post-synaptic states."""

    def __init__(self, jparams):
        super(InstarD, self).__init__()
        self.dt = jparams['dt']

    def forward(self, w, s):
        coeff = 1 / s[-1].size(0)
        for i in range(len(w)):
            V = coeff * torch.mm(s[i].T, s[i + 1])
            w[i] = (w[i] + self.dt * (V - w[i])).clamp(0, 1)
        return w


class MacDynamic(nn.Module):
    """Network with static weights M (self.W) and dynamic weights w (self.dyn).

    Layer order: s[0] = output, ..., s[-1] = input (clamped to the image).
    mac = 'total'   (Variant A): the neurons use the effective weight M + g * w, g = w_scale
    mac = 'partial' (Variant B): the neurons use M only; w is only used to compute the weight update.
    """

    def __init__(self, jparams, rho, rhop):
        super(MacDynamic, self).__init__()
        self.fcLayers = jparams['fcLayers']
        self.dt = jparams['dt']
        self.T = jparams['T']
        self.Kmax = jparams['Kmax']
        self.beta = jparams['beta']
        self.w_scale = jparams['w_scale']
        self.mac = jparams['mac']
        if self.mac not in ('total', 'partial'):
            raise ValueError(f"mac = '{self.mac}' is not defined (use 'total' = Variant A or 'partial' = Variant B)")
        self.grad_clip = jparams.get('grad_clip', None)              # per-layer norm clipping of every update (None = off)
        self.grad_clip_value = jparams.get('grad_clip_value', None)  # per-layer element-wise bound (None = off)
        self.rho = rho
        self.rhop = rhop

        if jparams['device'] >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(jparams['device']))
        else:
            self.device = torch.device("cpu")

        # static weights M: W[i] connects layer i and layer i + 1 (no bias)
        W = nn.ModuleList(None)
        for i in range(len(self.fcLayers) - 1):
            W.extend([nn.Linear(self.fcLayers[i + 1], self.fcLayers[i], bias=False)])
        self.W = W
        self.to(self.device)

        self.instarD = InstarD(jparams)
        # The values of self.dyn are overwritten by init_w() at every batch. 
        dyn = []
        for i in range(len(self.fcLayers) - 1):
            w = torch.empty(self.fcLayers[i], self.fcLayers[i + 1], device=self.device, requires_grad=False)
            nn.init.xavier_uniform_(w, gain=0.5)
            dyn.append(w)
        self.dyn = dyn

    def init_state(self, data):
        state = []
        with torch.no_grad():
            for layer in range(len(self.fcLayers) - 1):
                state.append(torch.zeros(data.size(0), self.fcLayers[layer], device=data.device, requires_grad=False))
            state.append(data.float())
        return state

    def init_w(self):
        """Reset the dynamic weights to 0 at the start of every mini-batch."""
        for i in range(len(self.dyn)):
            self.dyn[i] = torch.zeros(self.fcLayers[i], self.fcLayers[i + 1], device=self.device, requires_grad=False)

    def update_W(self, w_free, w_nudge):
        """Gradient estimate -(w_nudge - w_free) / beta, optionally clipped, handed to the optimiser."""
        coeff = 1 / self.beta
        for i in range(len(self.W)):
            grad = -coeff * (w_nudge[i] - w_free[i])
            if self.grad_clip is not None:
                norm = grad.norm()
                grad = grad * torch.clamp(self.grad_clip / (norm + 1e-12), max=1.0)
            if self.grad_clip_value is not None:
                grad = grad.clamp(-self.grad_clip_value[i], self.grad_clip_value[i])
            self.W[i].weight.grad = grad

    def update_s(self, s, target=None):
        """One step of the neuron states; target is not None during the nudge phase."""
        if self.mac == 'total':
            if target is None:
                s[0] = self.rho(self.rhop(s[0]) * torch.mm(s[1], self.w_scale * self.dyn[0].T + self.W[0].weight.T))
            else:
                s[0] = self.rho(self.rhop(s[0]) * torch.mm(s[1], self.w_scale * self.dyn[0].T + self.W[0].weight.T)
                                + self.beta * (target - s[0]))
            for i in range(1, len(s) - 1):
                s[i] = self.rho(self.rhop(s[i]) * (torch.mm(s[i + 1], self.w_scale * self.dyn[i].T + self.W[i].weight.T)
                                                   + torch.mm(s[i - 1], self.w_scale * self.dyn[i - 1] + self.W[i - 1].weight)))
        else:
            if target is None:
                s[0] = self.rho(self.rhop(s[0]) * torch.mm(s[1], self.W[0].weight.T))
            else:
                s[0] = self.rho(self.rhop(s[0]) * torch.mm(s[1], self.W[0].weight.T) + self.beta * (target - s[0]))
            for i in range(1, len(s) - 1):
                s[i] = self.rho((torch.mm(s[i + 1], self.W[i].weight.T)
                                 + torch.mm(s[i - 1], self.W[i - 1].weight)) * self.rhop(s[i]))
        return s

    def forward(self, s, target):
        """Free phase (T steps, target None) or nudge phase (Kmax steps); the dynamic weights evolve with the states.
        Returns the states and a copy of the dynamic weights at the end of the phase."""
        steps = self.T if target is None else self.Kmax
        with torch.no_grad():
            for _ in range(steps):
                s = self.update_s(s, target=target)
                self.dyn = self.instarD(self.dyn, s)
        return s, [item.clone() for item in self.dyn]

"""Fast, exact re-implementation of the Quantum Visual Fields (QVF) image model.

Reference: Wang, Theobalt, Golyanik, "Quantum Visual Fields with Neural Amplitude
Encoding", NeurIPS 2025, arXiv 2508.10900. Official code:
github.com/shutengW/Quantum-Visual-Fields-with-Neural-Amplitude-Encoding
(commit cb647ef, src/Images/QVF_for_image.py).

The official code simulates the circuit with PennyLane, gate by gate, for every
pixel in the batch. Here the circuit is simulated exactly in plain PyTorch, using
two facts about the QVF ansatz:

1. It contains only RY and controlled-RY gates, so it is a real orthogonal matrix.
2. Its angles do not depend on the input. The input enters only through the
   initial amplitudes. So the whole circuit is one 2^n x 2^n matrix per training
   step, shared by every pixel.

The matrix is rebuilt from the angles at every forward pass, so gradients reach
the angles through ordinary backpropagation. Module and parameter names mirror
the official code, so an official state_dict loads directly.

Conventions follow PennyLane: wire 0 is the most significant bit, and
RY(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]].
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def positional_embedding(x, num_frequencies, include_input=True, scale=1.0):
    """Same layout as the official code: [x, sin(x * 2^k), cos(x * 2^k)], k = 0..L-1.

    scale multiplies the frequencies. The official code uses 1. The paper's Eq. (1) has a
    factor of pi, sin(2^k * pi * x), which is scale=math.pi.
    """
    if num_frequencies <= 0:
        raise ValueError("num_frequencies must be a positive integer")
    parts = [x] if include_input else []
    freq_bands = scale * 2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype)
    x_expanded = x.unsqueeze(-1) * freq_bands
    parts.extend([torch.sin(x_expanded), torch.cos(x_expanded)])
    return torch.cat([p.reshape(*x.shape[:-1], -1) for p in parts], dim=-1)


def find_qubit_num(feature_dim):
    if feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    return (feature_dim - 1).bit_length()


class MLPEmbedding(nn.Module):
    """Classical network that outputs the amplitudes ("neural amplitude encoding").

    With amplitude_output=True it ends in a Boltzmann (softmax) map and a square
    root, as in the official code. With amplitude_output=False it is a plain MLP
    whose last layer gives the output directly, used for the classical baseline.

    activation="relu" is the official code. activation="sin" is my reading of the
    paper's "+Sin" variants (SIREN layers with the standard SIREN initialisation);
    the released code does not contain it.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, skip=False,
                 stabilization=False, activation="relu", amplitude_output=True, w0=30.0):
        super().__init__()
        self.skip = skip
        self.stabilization = stabilization
        self.input_dim = input_dim
        self.activation = activation
        self.amplitude_output = amplitude_output
        self.w0 = w0

        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            if skip and i == len(dims) - 3:
                self.layers.append(nn.Linear(dims[i], dims[i + 1] - input_dim))
            else:
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == "sin":
            with torch.no_grad():
                for i, layer in enumerate(self.layers):
                    fan_in = layer.weight.shape[1]
                    bound = 1.0 / fan_in if i == 0 else math.sqrt(6.0 / fan_in) / w0
                    layer.weight.uniform_(-bound, bound)
        elif activation != "relu":
            raise ValueError(f"unknown activation {activation}")

    @staticmethod
    def boltzmann(x, temperature, stabilization):
        if stabilization:
            return torch.softmax(x / temperature, dim=-1)
        max_x = torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp((x - max_x) / temperature)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

    def forward(self, x, temperature=1.0):
        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x) if self.activation == "relu" else torch.sin(self.w0 * x)
                if self.skip and i == len(self.layers) - 2:
                    x = torch.cat([residual, x], dim=-1)
        if not self.amplitude_output:
            return x
        return torch.sqrt(self.boltzmann(x, temperature, self.stabilization))


def _apply_ry(state, theta, axis):
    """Apply RY(theta) along one tensor axis of size 2."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    a0, a1 = state.unbind(dim=axis)
    return torch.stack((c * a0 - s * a1, s * a0 + c * a1), dim=axis)


def _apply_cry(state, theta, control_axis, target_axis):
    """Apply RY(theta) on target_axis, only where the control qubit is 1."""
    s0, s1 = state.unbind(dim=control_axis)
    t = target_axis if target_axis < control_axis else target_axis - 1
    return torch.stack((s0, _apply_ry(s1, theta, t)), dim=control_axis)


class FastQuantumLayer(nn.Module):
    """Exact simulation of the official QuantumLayer without PennyLane.

    Parameter shapes and initialisation match the official code:
      identity block: weights (2n, J), cond (2n, n, J); the second half starts as a
                      copy of the first, so each block starts as the identity.
      otherwise:      weights (n, J),  cond (n, n, J), Gaussian with gaussian_std.
    The diagonal entries cond[i, i] are unused, as in the official code.

    Note on the official code: its rotation layers index a 2D tensor with three
    indices (weights[:, :, r]), which raises an IndexError. The intended reading,
    used here, is weights[:, r]: one RY angle per qubit for repetition r.
    """

    cpu_matrix_max_qubits = 6

    def __init__(self, n_qubits, repetitions, use_identity_block, gaussian_std=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.repetitions = repetitions
        self.use_identity_block = use_identity_block

        if use_identity_block:
            w = torch.rand(n_qubits, repetitions) * 2 * math.pi
            c = torch.rand(n_qubits, n_qubits, repetitions) * 2 * math.pi
            self.weights = nn.Parameter(torch.cat([w, w], dim=0))
            self.cond = nn.Parameter(torch.cat([c, c], dim=0))
        else:
            self.weights = nn.Parameter(torch.normal(0.0, gaussian_std, size=(n_qubits, repetitions)))
            self.cond = nn.Parameter(torch.normal(0.0, gaussian_std, size=(n_qubits, n_qubits, repetitions)))

        # signs[k, q] = +1 if qubit q of basis state k is 0, else -1 (wire 0 = most significant bit).
        k = torch.arange(2 ** n_qubits).unsqueeze(1)
        shifts = torch.arange(n_qubits - 1, -1, -1).unsqueeze(0)
        self.register_buffer("signs", 1.0 - 2.0 * ((k >> shifts) & 1).double(), persistent=False)

    def _run_gates(self, state, weights=None, cond=None):
        """Apply the ansatz to a batch of states of shape (K, 2, ..., 2). Wire q is axis q + 1."""
        weights = self.weights if weights is None else weights
        cond = self.cond if cond is None else cond
        n = self.n_qubits
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        for r in range(self.repetitions):
            for i, j in pairs:                                   # entangle
                state = _apply_cry(state, cond[i, j, r], i + 1, j + 1)
            for q in range(n):                                   # rotations
                state = _apply_ry(state, weights[q, r], q + 1)
            if self.use_identity_block:
                for q in reversed(range(n)):                     # adjoint(rotations)
                    state = _apply_ry(state, -weights[n + q, r], q + 1)
                for i, j in reversed(pairs):                     # adjoint(entangle)
                    state = _apply_cry(state, -cond[n + i, j, r], i + 1, j + 1)
        return state

    def circuit_matrix(self):
        """Matrix M of shape (2^n, 2^n) with final_state = amplitudes @ M. M is the transpose of the unitary.

        Up to cpu_matrix_max_qubits the matrix is built on the CPU in double precision and then
        moved to the parameters' device and dtype. The build is a few thousand operations on tiny
        tensors, which are cheaper on the CPU (about 2x faster per training step at 5 qubits on
        an RTX 4090, and slower from 8 qubits on). Gradients flow through the device move.
        """
        n, dim = self.n_qubits, 2 ** self.n_qubits
        weights, cond = self.weights, self.cond
        if n <= self.cpu_matrix_max_qubits:
            weights, cond = weights.to("cpu", torch.float64), cond.to("cpu", torch.float64)
        basis = torch.eye(dim, dtype=weights.dtype, device=weights.device)
        matrix = self._run_gates(basis.reshape(dim, *([2] * n)), weights, cond).reshape(dim, dim)
        return matrix.to(self.weights.device, self.weights.dtype)

    def forward(self, amplitudes, per_sample=False):
        """Return the Pauli-Z expectation of every qubit, shape (batch, n).

        per_sample=True applies the gates to every input state separately. It gives
        the same numbers and exists only to test the single-matrix shortcut.
        """
        n, dim = self.n_qubits, 2 ** self.n_qubits
        if amplitudes.shape[-1] < dim:                           # pad_with=0.0 in the official code
            amplitudes = F.pad(amplitudes, (0, dim - amplitudes.shape[-1]))
        if per_sample:
            final = self._run_gates(amplitudes.reshape(-1, *([2] * n))).reshape(-1, dim)
        else:
            final = amplitudes @ self.circuit_matrix()
        return (final * final) @ self.signs.to(final.dtype)


def parity_decode(x, channels):
    """Official read-out: sum the Z expectations in groups of `channels` (zero padded)."""
    dim = x.shape[-1]
    if dim < channels:
        raise ValueError("the number of wires must be at least the number of channels")
    if dim % channels != 0:
        x = F.pad(x, (0, channels - dim % channels))
    grouped = torch.stack(torch.split(x, channels, dim=-1), dim=-1)
    normalization = dim // channels if dim // channels > 0 else 1
    return grouped.sum(dim=-1) / normalization


class QVFImage(nn.Module):
    """QVF image model: MLP amplitude encoding, circuit, Z read-out. Mirrors the official ImageRegressor."""

    def __init__(self, cfg):
        super().__init__()
        coord_dim = 2 + 4 * cfg["L_embed"] if cfg["L_embed"] else 2
        self.embedder = MLPEmbedding(
            input_dim=coord_dim + cfg["latent_size"],
            hidden_dim=cfg["hid_dim"],
            output_dim=cfg["out_dim"],
            num_hidden_layers=cfg["num_hidden_layers_energy"],
            skip=cfg["skip_connection"],
            activation=cfg.get("activation", "relu"),
        )
        self.quantum = FastQuantumLayer(
            n_qubits=find_qubit_num(cfg["out_dim"]),
            repetitions=cfg["num_repetations"],
            use_identity_block=cfg["use_identity_block"],
            gaussian_std=cfg["gaussian_std"],
        )
        self.channels = cfg["output_channels"]
        self.use_parity = cfg["use_parity"]

    def forward(self, x, per_sample=False):
        z = self.quantum(self.embedder(x), per_sample=per_sample)
        return parity_decode(z, self.channels) if self.use_parity else z[:, :self.channels]


class MLPImage(nn.Module):
    """Classical baseline: the same MLP trunk, with the last layer giving the colour directly.

    This is my reading of the paper's "MLP+ReLU" and "MLP+Sin" baselines ("the only
    difference between the baselines and QVF is the presence of the ansatz"). The
    released code does not include the baseline.
    """

    def __init__(self, cfg):
        super().__init__()
        coord_dim = 2 + 4 * cfg["L_embed"] if cfg["L_embed"] else 2
        self.embedder = MLPEmbedding(
            input_dim=coord_dim + cfg["latent_size"],
            hidden_dim=cfg["hid_dim"],
            output_dim=cfg["output_channels"],
            num_hidden_layers=cfg["num_hidden_layers_energy"],
            skip=cfg["skip_connection"],
            activation=cfg.get("activation", "relu"),
            amplitude_output=False,
        )

    def forward(self, x):
        return self.embedder(x)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

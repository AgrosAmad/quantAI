"""Explicit RY/CNOT execution for Z4 at five or seven qubits."""
from collections import Counter
import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
import torch
from .spatial_models import SpatialField
from .feasibility import observable_rgb as five_wire_observable, shot_rgb


def observable_rgb(qubits):
    return five_wire_observable()[np.arange(2**qubits) >> (qubits - 5)]


def marginal_readout(probabilities):
    """Exactly marginalize unmeasured work wires; retain joint data-wire outcomes."""
    p = np.asarray(probabilities, dtype=np.float64)
    return p.reshape(len(p), 32, -1).sum(-1)


def shot_mse_moments(probabilities, targets, shots):
    """Exact per-coordinate mean/variance of RGB MSE from joint iid shots.

    Uses covariance and third/fourth central moments of the joint RGB outcome.
    This supplies an analytic Monte Carlo standard error, without estimating
    its acceptance threshold from only five measurement replicates.
    """
    p = marginal_readout(probabilities)
    obs = five_wire_observable()
    mean = p @ obs
    centered = obs[None] - mean[:, None]
    norm2 = (centered**2).sum(-1)
    covariance = np.einsum('ni,nic,nid->ncd', p, centered, centered)
    trace = np.trace(covariance, axis1=1, axis2=2)
    third = np.einsum('ni,nic,ni->nc', p, centered, norm2)
    fourth = (p * norm2**2).sum(-1)
    delta = mean - np.asarray(targets, dtype=np.float64)
    expected = ((delta**2).sum(-1) + trace/shots)/3
    variance = (4*np.einsum('nc,ncd,nd->n', delta, covariance, delta)/shots
        + 4*(delta*third).sum(-1)/shots**2
        + (fourth - trace**2 + 2*(shots-1)*(covariance**2).sum((1, 2)))/shots**3)/9
    assert variance.min() > -1e-12
    return expected, np.maximum(variance, 0.)


@torch.no_grad()
def inputs_and_probabilities(model, codes, coordinates):
    data = model.encoded(codes, coordinates)
    slot = isinstance(model, SpatialField)
    state = model.quantum.slot_state(data) if slot else data @ model.quantum.circuit_matrix()
    p = state.square().double()
    p /= p.sum(-1, keepdim=True)
    return ('slot' if slot else 'amplitude'), data.cpu().double().numpy(), p.cpu().numpy()


def primitive_ops(kind, data, quantum):
    n, ops = quantum.n_qubits, []
    if kind == 'amplitude':
        a = np.asarray(data, dtype=np.float64)
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        if np.any(a < 0) or a.shape[1] != 2**n:
            raise ValueError('Expected normalized positive real amplitudes of the declared size')
        reverse = list(range(n))[::-1]
        for k in range(n, 0, -1):
            squares = a.reshape(len(a), 2**(n-k), 2**k)**2
            total = squares.sum(-1)
            ratio = np.divide(squares[..., 2**(k-1):].sum(-1), total,
                out=np.zeros_like(total), where=total > 0)
            theta = compute_theta(2 * np.arcsin(np.sqrt(np.clip(ratio, 0, 1))))
            controls, target = reverse[k:], reverse[k-1]
            if not controls:
                ops.append(qml.RY(theta[:, 0], wires=target))
            else:
                code = gray_code(len(controls))
                for i in range(len(code)):
                    ops.append(qml.RY(theta[:, i], wires=target))
                    bit = (int(code[i], 2) ^ int(code[(i+1) % len(code)], 2)).bit_length()-1
                    ops.append(qml.CNOT(wires=[controls[bit], target]))
    elif kind == 'slot':
        if data.shape[1:] != (5, 5):
            raise ValueError('Exactly five data slots at either qubit count')
        ops.extend(qml.RY(np.pi/2, wires=w) for w in range(5))
    else:
        raise ValueError(kind)
    weights = quantum.weights.detach().cpu().double().numpy()
    filters = quantum.filters.detach().cpu().double().numpy()
    for layer in range(5):
        if kind == 'slot':
            ops.extend(qml.RY(data[:, layer, w], wires=w) for w in range(5))
        for a, b in quantum.pairs:
            x = filters[layer, 0]
            ops.extend((qml.RY(x[0], wires=a), qml.RY(x[1], wires=b)))
            ops.extend(qml.CRY(x[2], wires=[a, b]).decomposition())
            ops.extend(qml.CRY(x[3], wires=[b, a]).decomposition())
        ops.extend(qml.RY(weights[w, layer], wires=w) for w in range(n))
    return [(op.name, tuple(int(w) for w in op.wires), np.asarray(op.data[0]) if op.data else None) for op in ops]


def gate_resources(ops, qubits):
    def depth(sequence):
        available = [0] * qubits
        for _, wires, _ in sequence:
            end = max(available[w] for w in wires)+1
            for w in wires:
                available[w] = end
        return max(available)
    routed = []
    for name, wires, angle in ops:
        if name != 'CNOT' or abs(wires[0]-wires[1]) == 1:
            routed.append((name, wires, angle)); continue
        a, b = wires
        direction = 1 if b > a else -1
        swaps = [(x, x+direction) for x in range(a, b-direction, direction)]
        for x, y in swaps:
            routed.extend([('CNOT', (x, y), None), ('CNOT', (y, x), None), ('CNOT', (x, y), None)])
        routed.append(('CNOT', (b-direction, b), None))
        for x, y in reversed(swaps):
            routed.extend([('CNOT', (x, y), None), ('CNOT', (y, x), None), ('CNOT', (x, y), None)])
    return {'all_to_all_primitives': dict(Counter(o[0] for o in ops)), 'all_to_all_depth': depth(ops),
        'line_restore_after_each_gate_primitives': dict(Counter(o[0] for o in routed)),
        'line_restore_after_each_gate_depth': depth(routed),
        'routing': 'Unoptimized SWAP routing, restores logical order after each nonlocal CNOT. Not a hardware-optimal lower bound. Noise diagnostic excludes routing.'}


def depolarize(rho, wires, probability, qubits):
    if probability == 0:
        return rho
    wires = list(wires)
    rest = [w for w in range(qubits) if w not in wires]
    order = wires + rest
    permutation = [0] + [w+1 for w in order] + [w+qubits+1 for w in order]
    inverse = np.argsort(permutation).tolist()
    d, r = 2**len(wires), 2**len(rest)
    matrix = rho.reshape(-1, *([2]*(2*qubits))).permute(permutation).reshape(-1, d, r, d, r)
    reduced = torch.einsum('baiaj->bij', matrix)
    mixed = torch.eye(d, device=rho.device, dtype=rho.dtype)[None, :, None, :, None] * reduced[:, None, :, None, :] / d
    mixed = mixed.reshape(-1, *([2]*(2*qubits))).permute(inverse).reshape_as(rho)
    alpha = probability * d*d / (d*d-1)
    return (1-alpha) * rho + alpha * mixed


@torch.no_grad()
def simulate_density(ops, batch, qubits, device='cpu', p1=0., p2=0., readout=0.):
    dim = 2**qubits
    rho = torch.zeros(batch, dim, dim, dtype=torch.float64, device=device)
    rho[:, 0, 0] = 1.
    indices = torch.arange(dim, device=device)
    for name, wires, angle in ops:
        if name == 'RY':
            angle = torch.as_tensor(np.array(angle, copy=True), dtype=rho.dtype, device=device)
            if angle.ndim == 0:
                angle = angle.expand(batch)
            c = (angle/2).cos().reshape(batch, *([1]*(2*qubits-1)))
            s = (angle/2).sin().reshape(batch, *([1]*(2*qubits-1)))
            state = rho.reshape(batch, *([2]*(2*qubits)))
            for axis in (wires[0]+1, wires[0]+qubits+1):
                a, b = state.unbind(axis)
                state = torch.stack((c*a-s*b, s*a+c*b), axis)
            rho = depolarize(state.reshape(batch, dim, dim), wires, p1, qubits)
        elif name == 'CNOT':
            a, b = wires
            permutation = indices ^ (((indices >> (qubits-1-a)) & 1) << (qubits-1-b))
            rho = depolarize(rho[:, permutation][:, :, permutation], wires, p2, qubits)
        else:
            raise ValueError(name)
    assert (rho.diagonal(dim1=1, dim2=2).sum(-1)-1).abs().max() < 1e-9
    p = rho.diagonal(dim1=1, dim2=2).clone()
    # Only wires actually read are flipped; work-wire readout is immaterial.
    for wire in range(5):
        p = (1-readout)*p + readout*p[:, indices ^ (1 << (qubits-1-wire))]
    assert p.min() > -1e-10
    p = p.clamp_min(0)
    return (p / p.sum(-1, keepdim=True)).cpu().numpy()

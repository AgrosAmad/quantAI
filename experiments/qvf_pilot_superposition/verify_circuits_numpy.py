"""Independent NumPy checks for the pilot circuits (no PyTorch needed).

Each model's fast simulation (the same tensor tricks used in the notebook) is compared
with a slow, explicit simulation that builds full gate matrices with Kronecker products.
Run:  python verify_circuits_numpy.py      Expected: every line ends with OK.
"""
import numpy as np

rng = np.random.default_rng(0)
I2 = np.eye(2); Z2 = np.diag([1., -1.])
TOL = 1e-10


def unbind(x, ax): return [np.take(x, i, axis=ax) for i in range(x.shape[ax])]

# ---------- fast simulator (mirror of the notebook) ----------
def ry(state, theta, wire):
    axis = wire + 1; theta = np.asarray(theta, float)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if theta.ndim == 1:
        shape = (-1,) + (1,) * (state.ndim - 2); c = c.reshape(shape); s = s.reshape(shape)
    a0, a1 = unbind(state, axis)
    return np.stack((c * a0 - s * a1, s * a0 + c * a1), axis)

def cry(state, theta, control, target):
    s0, s1 = unbind(state, control + 1); t = target if target < control else target - 1
    return np.stack((s0, ry(s1, theta, t)), control + 1)

def ry_branch(state, theta, wire):
    axis = 2 + wire; c, s = np.cos(theta / 2), np.sin(theta / 2)
    shape = theta.shape + (1,) * (state.ndim - 3); c = c.reshape(shape); s = s.reshape(shape)
    a0, a1 = unbind(state, axis)
    return np.stack((c * a0 - s * a1, s * a0 + c * a1), axis)

def ry_mat(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack((np.stack((c, -s), -1), np.stack((s, c), -1)), -2)

def upload_unitary(theta):           # theta (..., 5) -> (..., 32, 32) = RY_0 (x) ... (x) RY_4
    V = ry_mat(theta[..., 0])
    for w in range(1, 5):
        R = ry_mat(theta[..., w])
        V = np.einsum('...ab,...cd->...acbd', V, R).reshape(*theta.shape[:-1], V.shape[-2] * 2, V.shape[-1] * 2)
    return V

# ---------- explicit reference ----------
def RY(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]])
def op1(U, w, n):
    m = np.array([[1.]])
    for q in range(n): m = np.kron(m, U if q == w else I2)
    return m
def ctrl(U, c, t, n):
    P0 = np.diag([1., 0]); P1 = np.diag([0, 1.]); A = np.array([[1.]]); B = np.array([[1.]])
    for q in range(n):
        A = np.kron(A, P0 if q == c else I2); B = np.kron(B, P1 if q == c else (U if q == t else I2))
    return A + B

def report(name, err):
    print(f'{name:60s} max diff {err:.2e}  {"OK" if err < TOL else "FAIL"}')
    return err < TOL

ok = True
R = 5; N = 3

# ===== 1. Painter: 7 qubits, quantum and no_entanglement =====
n = 7; pairs = [(2, i) for i in range(n) if i != 2]
filters = rng.normal(size=(R, 4)); weights = rng.normal(size=(n, R)); a = rng.normal(size=(N, R, 5))
signs7 = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
for mode in ('quantum', 'no_entanglement'):
    def block(r):
        st = np.eye(2 ** n).reshape(2 ** n, *[2] * n); f = filters[r]
        for x, y in pairs:
            st = ry(st, f[0], x); st = ry(st, f[1], y)
            if mode == 'quantum': st = cry(st, f[2], x, y); st = cry(st, f[3], y, x)
        for q in range(n): st = ry(st, weights[q, r], q)
        return st.reshape(2 ** n, 2 ** n)
    state = np.zeros((N, 2 ** n)); state[:, 0] = 1; state = state.reshape(N, *[2] * n)
    for w in range(5): state = ry(state, np.pi / 2, w)
    for r in range(R):
        for w in range(5): state = ry(state, a[:, r, w], w)
        state = (state.reshape(N, -1) @ block(r)).reshape(N, *[2] * n)
    z = (state.reshape(N, -1) ** 2) @ signs7
    err = 0
    for i in range(N):
        psi = np.zeros(2 ** n); psi[0] = 1
        for w in range(5): psi = op1(RY(np.pi / 2), w, n) @ psi
        for r in range(R):
            for w in range(5): psi = op1(RY(a[i, r, w]), w, n) @ psi
            f = filters[r]
            for x, y in pairs:
                psi = op1(RY(f[0]), x, n) @ psi; psi = op1(RY(f[1]), y, n) @ psi
                if mode == 'quantum': psi = ctrl(RY(f[2]), x, y, n) @ psi; psi = ctrl(RY(f[3]), y, x, n) @ psi
            for q in range(n): psi = op1(RY(weights[q, r]), q, n) @ psi
        zr = np.array([psi @ op1(Z2, q, n) @ psi for q in range(n)])
        err = max(err, np.abs(zr - z[i]).max())
    ok &= report(f'7-qubit {mode}: fast vs explicit Z expectations', err)
p = (state.reshape(N, -1) ** 2)[0].reshape([2] * n); p02 = p.sum(axis=(1, 3, 4, 5, 6))
ok &= report('no_entanglement gives a product state (qubits 0 and 2 independent)', np.abs(p02 - np.outer(p02.sum(1), p02.sum(0))).max())

# ===== 2. Superposition models: 5 content + 3 address qubits =====
CTRL = (0, 2, 4); n8 = 8; pairs5 = [(2, i) for i in range(5) if i != 2]
centre = rng.normal(size=(N, R, 5)); nbr = rng.normal(size=(N, 8, R, 5))
filters = rng.normal(size=(R, 4)); weights = rng.normal(size=(5, R))
signs5 = 1 - 2 * ((np.arange(32)[:, None] >> np.arange(4, -1, -1)) & 1)

def block5(r):
    st = np.eye(32).reshape(32, *[2] * 5); f = filters[r]
    for x, y in pairs5:
        st = ry(st, f[0], x); st = ry(st, f[1], y); st = cry(st, f[2], x, y); st = cry(st, f[3], y, x)
    for q in range(5): st = ry(st, weights[q, r], q)
    return st.reshape(32, 32)

def mix_matrix(alpha, beta, r):       # row convention, as in the notebook
    st = np.eye(256).reshape(256, *[2] * 8)
    for j in range(3): st = ry(st, alpha[r, j], j); st = cry(st, beta[r, j], 3 + CTRL[j], j)
    return st.reshape(256, 256)

def coherent_fast(alpha, beta):
    state = np.full((N, 8, 32), 1 / 16.).reshape(N, 8, *[2] * 5)
    for r in range(R):
        for w in range(5):
            state = ry(state, centre[:, r, w], 1 + w); state = ry_branch(state, nbr[:, :, r, w], w)
        state = (state.reshape(N * 8, 32) @ block5(r)).reshape(N, 8, *[2] * 5)
        if r < R - 1:
            full = state.reshape(N, *[2] * 8)
            for j in range(3): full = ry(full, alpha[r, j], j); full = cry(full, beta[r, j], 3 + CTRL[j], j)
            state = full.reshape(N, 8, *[2] * 5)
    return (state.reshape(N, 8, 32) ** 2).sum(1)                       # content probabilities (N, 32)

def dephased_fast(alpha, beta):
    theta = centre[:, None] + nbr                                       # (N, 8, R, 5)
    rho = np.full((N, 8, 32, 32), 1 / 256.)
    for r in range(R):
        V = upload_unitary(theta[:, :, r])
        rho = V @ rho @ np.swapaxes(V, -1, -2)
        B = block5(r); rho = B.T @ rho @ B
        if r < R - 1:
            A = mix_matrix(alpha, beta, r).T.reshape(8, 32, 8, 32)
            new = 0
            for k in range(8):
                Ak = A[:, :, k, :]
                X = np.einsum('jab,nbc->njac', Ak, rho[:, k])
                new = new + np.einsum('njac,jdc->njad', X, Ak)
            rho = new
    return np.diagonal(rho, axis1=-2, axis2=-1).sum(1)

def gates_round(i, r, alpha, beta):
    """explicit list of 256x256 unitaries for round r (uploads, block, mixing)"""
    Us = []
    for w in range(5):
        Us.append(op1(RY(centre[i, r, w]), 3 + w, n8))
        P = lambda k: np.kron(np.diag(np.eye(8)[k]), np.eye(32))
        Us.append(sum(P(k) @ op1(RY(nbr[i, k, r, w]), 3 + w, n8) for k in range(8)))
    f = filters[r]
    for x, y in pairs5:
        Us += [op1(RY(f[0]), 3 + x, n8), op1(RY(f[1]), 3 + y, n8), ctrl(RY(f[2]), 3 + x, 3 + y, n8), ctrl(RY(f[3]), 3 + y, 3 + x, n8)]
    for q in range(5): Us.append(op1(RY(weights[q, r]), 3 + q, n8))
    mix = []
    if r < R - 1:
        for j in range(3): mix += [op1(RY(alpha[r, j]), j, n8), ctrl(RY(beta[r, j]), 3 + CTRL[j], j, n8)]
    return Us, mix

def dephase_address(rho):              # remove all coherence between different address values
    out = np.zeros_like(rho)
    for k in range(8):
        s = slice(32 * k, 32 * k + 32); out[s, s] = rho[s, s]
    return out

alpha = rng.normal(size=(R - 1, 3)); beta = rng.normal(size=(R - 1, 3))
pc = coherent_fast(alpha, beta); pd = dephased_fast(alpha, beta)
e_c = e_d = 0
for i in range(N):
    psi = np.full(256, 1 / 16.)
    rho = dephase_address(np.outer(psi, psi))
    for r in range(R):
        Us, mix = gates_round(i, r, alpha, beta)
        for U in Us + mix: psi = U @ psi
        for U in Us: rho = U @ rho @ U.T
        if mix:
            for U in mix: rho = U @ rho @ U.T
            rho = dephase_address(rho)
    e_c = max(e_c, np.abs((psi ** 2).reshape(8, 32).sum(0) - pc[i]).max())
    e_d = max(e_d, np.abs(np.diag(rho).reshape(8, 32).sum(0) - pd[i]).max())
ok &= report('superposition (coherent): fast vs explicit 256x256 simulation', e_c)
ok &= report('superposition_dephased: fast vs explicit density matrix + dephasing', e_d)
ok &= report('dephased model keeps total probability = 1', np.abs(pd.sum(1) - 1).max())
z0 = np.zeros((R - 1, 3))
ok &= report('no mixing (alpha=beta=0): coherent and dephased are identical', np.abs(coherent_fast(z0, z0) - dephased_fast(z0, z0)).max())
diff = np.abs(pc - pd).max()
print(f'{"with mixing: coherent and dephased differ (interference is real)":60s} max diff {diff:.2e}  {"OK" if diff > 1e-3 else "FAIL"}')
ok &= diff > 1e-3
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')

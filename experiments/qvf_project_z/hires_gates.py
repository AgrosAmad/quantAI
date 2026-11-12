"""Independent primitive execution and shared-bitstring readout for Z6."""
import functools
import math
import numpy as np
import pennylane as qml
import torch

from .spatial_gates import primitive_ops as old_ops, gate_resources, depolarize


def primitives(head, data, amplitude=False):
    if amplitude:
        assert head.extra=='none'
        return old_ops('amplitude',data,head)
    operations=[qml.RY(np.pi/2,wires=i) for i in range(5)]
    weights=head.weights.detach().cpu().double().numpy()
    filters=head.filters.detach().cpu().double().numpy()
    if head.extra!='none':
        single=head.extra_single.detach().cpu().double().numpy()
        pair=head.extra_pair.detach().cpu().double().numpy()
    for layer in range(5):
        operations.extend(qml.RY(data[:,layer,i],wires=i) for i in range(5))
        if head.extra!='none':
            gate=qml.RZ if head.extra=='phase' else qml.RY
            operations.extend(gate(single[layer,i],wires=i) for i in range(5))
        for a,b in head.pairs:
            f=filters[layer,0]
            operations.extend((qml.RY(f[0],wires=a),qml.RY(f[1],wires=b)))
            operations.extend(qml.CRY(f[2],wires=[a,b]).decomposition())
            operations.extend(qml.CRY(f[3],wires=[b,a]).decomposition())
            if head.extra=='phase':
                operations.extend((qml.CNOT(wires=[a,b]),qml.RZ(pair[layer],wires=b),qml.CNOT(wires=[a,b])))
            elif head.extra=='real':operations.extend(qml.CRY(pair[layer],wires=[a,b]).decomposition())
        operations.extend(qml.RY(weights[i,layer],wires=i) for i in range(head.n_qubits))
    return [(op.name,tuple(int(w) for w in op.wires),np.asarray(op.data[0]) if op.data else None) for op in operations]


def pennylane_state(head, data, amplitude=False, parameters=None):
    """Native CRY/IsingZZ QNode, independent of the matrix and primitive implementations."""
    p={k:v.to('cpu',torch.float64) for k,v in head.named_parameters()} if parameters is None else parameters
    x=torch.as_tensor(data,dtype=torch.float64)
    @qml.qnode(qml.device('default.qubit',wires=head.n_qubits),interface='torch',diff_method='backprop')
    def circuit():
        if amplitude:qml.AmplitudeEmbedding(x,wires=range(head.n_qubits),normalize=True)
        else:
            for wire in range(5):qml.RY(torch.pi/2,wires=wire)
        for layer in range(5):
            if not amplitude:
                for wire in range(5):qml.RY(x[layer,wire],wires=wire)
            if head.extra!='none':
                gate=qml.RZ if head.extra=='phase' else qml.RY
                for wire in range(5):gate(p['extra_single'][layer,wire],wires=wire)
            for a,b in head.pairs:
                f=p['filters'][layer,0]
                qml.RY(f[0],wires=a);qml.RY(f[1],wires=b)
                qml.CRY(f[2],wires=[a,b]);qml.CRY(f[3],wires=[b,a])
                if head.extra=='phase':qml.IsingZZ(p['extra_pair'][layer],wires=[a,b])
                elif head.extra=='real':qml.CRY(p['extra_pair'][layer],wires=[a,b])
            for wire in range(head.n_qubits):qml.RY(p['weights'][wire,layer],wires=wire)
        return qml.state()
    return circuit()


@torch.no_grad()
def density(ops, batch, qubits, device='cpu', p1=0., p2=0., readout=0.):
    d=2**qubits;rho=torch.zeros(batch,d,d,dtype=torch.complex128,device=device);rho[:,0,0]=1
    indices=torch.arange(d,device=device)
    for name,wires,angle in ops:
        if name=='RY':
            theta=torch.as_tensor(np.array(angle,copy=True),dtype=torch.float64,device=device)
            if theta.ndim==0:theta=theta.expand(batch)
            c=(theta/2).cos().reshape(batch,*([1]*(2*qubits-1)))
            s=(theta/2).sin().reshape(batch,*([1]*(2*qubits-1)))
            state=rho.reshape(batch,*([2]*(2*qubits)))
            for axis in (wires[0]+1,wires[0]+qubits+1):
                a,b=state.unbind(axis);state=torch.stack((c*a-s*b,s*a+c*b),axis)
            rho=depolarize(state.reshape(batch,d,d),wires,p1,qubits)
        elif name=='RZ':
            theta=torch.as_tensor(np.array(angle,copy=True),dtype=torch.float64,device=device)
            if theta.ndim==0:theta=theta.expand(batch)
            signs=1.-2.*((indices>>(qubits-1-wires[0]))&1)
            phase=torch.exp(-.5j*theta[:,None]*signs[None])
            rho=depolarize(phase[:,:,None]*rho*phase.conj()[:,None,:],wires,p1,qubits)
        elif name=='CNOT':
            a,b=wires;perm=indices^(((indices>>(qubits-1-a))&1)<<(qubits-1-b))
            rho=depolarize(rho[:,perm][:,:,perm],wires,p2,qubits)
        else:raise ValueError(name)
    diagonal=rho.diagonal(dim1=1,dim2=2)
    assert diagonal.imag.abs().max()<1e-9
    p=diagonal.real.clone();assert (p.sum(-1)-1).abs().max()<1e-9 and p.min()>-1e-9
    for wire in range(5):p=(1-readout)*p+readout*p[:,indices^(1<<(qubits-1-wire))]
    p=p.clamp_min(0);return (p/p.sum(-1,keepdim=True)).cpu().numpy()


SIGNS=1.-2.*((np.arange(32)[:,None]>>np.arange(4,-1,-1))&1)
PAIRS=(0,1,3,4)
JOINT=np.concatenate((SIGNS,SIGNS[:,2,None]*SIGNS[:,PAIRS]),axis=1)
RGB=np.stack((SIGNS[:,0]+SIGNS[:,3],SIGNS[:,1]+SIGNS[:,4],SIGNS[:,2]),axis=1)


def marginal(prob):
    p=np.asarray(prob,dtype=np.float64).reshape(len(prob),32,-1).sum(-1)
    return p/p.sum(-1,keepdims=True)


def readout_numpy(model, probabilities):
    ps=[marginal(p) for p in probabilities]
    if hasattr(model,'readout'):
        p=ps[0];z=p@SIGNS
        features=p@JOINT if model.cfg.architecture=='joint_statistics' else np.concatenate((z,z[:,2,None]*z[:,PAIRS]),1)
        return features@model.readout.weight.detach().cpu().double().numpy().T+model.readout.bias.detach().cpu().double().numpy()
    return np.mean([p@RGB for p in ps],axis=0)


def sample_probabilities(p, shots, seed):
    p=marginal(p);rng=np.random.default_rng(seed)
    counts=np.zeros_like(p);remaining=np.full(len(p),shots,dtype=np.int64);mass=np.ones(len(p))
    for i in range(31):
        ratio=np.divide(p[:,i],mass,out=np.zeros_like(mass),where=mass>1e-15)
        count=rng.binomial(remaining,np.clip(ratio,0,1));counts[:,i]=count
        remaining-=count;mass-=p[:,i]
    counts[:,-1]=remaining
    return counts/shots


def sampled_readout(model, probs, shots, seed):
    # Independent measurement randomness for independent circuit heads.
    samples=[sample_probabilities(p,shots,seed+100003*i) for i,p in enumerate(probs)]
    return readout_numpy(model,samples)


@functools.lru_cache(None)
def partitions(size):
    if size==0:return ((),)
    result=[]
    for previous in partitions(size-1):
        result.append(previous+((size-1,),))
        for i in range(len(previous)):
            result.append(previous[:i]+(previous[i]+(size-1,),)+previous[i+1:])
    return tuple(result)


def expected_mse(model, probs, target, shots):
    """Exact finite-shot MSE expectation, including marginal-product estimator bias."""
    ps=[marginal(p) for p in probs];target=np.asarray(target,dtype=np.float64)
    if model.cfg.architecture!='marginal_products':
        if hasattr(model,'readout'):
            w=model.readout.weight.detach().cpu().double().numpy()
            b=model.readout.bias.detach().cpu().double().numpy()
            obs=JOINT@w.T+b
        else:obs=RGB
        means=[p@obs for p in ps];mean=np.mean(means,0)
        variance=sum(p@(obs**2)-mu**2 for p,mu in zip(ps,means))/len(ps)**2
        return ((mean-target)**2+variance/shots).mean(-1)
    p=ps[0];w=model.readout.weight.detach().cpu().double().numpy();bias=model.readout.bias.detach().cpu().double().numpy()
    terms=[()]+[(i,) for i in range(5)]+[(2,i) for i in PAIRS]
    coefficients=np.concatenate(((bias[None]-target)[:,:,None],np.broadcast_to(w[None],(len(p),3,9))),axis=2)
    single_moments={0:np.ones(len(p))};cache={():np.ones(len(p))}
    def moment(indices):
        indices=tuple(sorted(indices))
        if indices in cache:return cache[indices]
        k=len(indices);value=np.zeros(len(p))
        for blocks in partitions(k):
            b=len(blocks)
            if b>shots:continue
            factor=math.prod(range(shots-b+1,shots+1))/shots**k
            product=np.ones(len(p))
            for block in blocks:
                mask=0
                for i in block:mask^=1<<indices[i]
                if mask not in single_moments:
                    obs=np.prod(SIGNS[:,[i for i in range(5) if mask>>i&1]],axis=1)
                    single_moments[mask]=p@obs
                product*=single_moments[mask]
            value+=factor*product
        cache[indices]=value;return value
    result=np.zeros((len(p),3))
    for i,left in enumerate(terms):
        for j,right in enumerate(terms):
            result+=coefficients[:,:,i]*coefficients[:,:,j]*moment(left+right)[:,None]
    assert result.min()>-1e-9
    return np.maximum(result,0).mean(-1)

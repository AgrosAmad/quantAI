"""Independent native circuits, batched primitive compilation and Z7 noise."""
import math
import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta,gray_code
import torch
from .spatial_gates import depolarize


def native_qnode(model,data,dephase=False):
    if model.variant=='classical_attention16':raise ValueError('Use native seven-qubit heads for fusion')
    inputs=[torch.as_tensor(x,dtype=torch.float64) for x in data]
    @qml.qnode(qml.device('default.qubit',wires=9),interface='torch',diff_method='backprop')
    def circuit():
        qml.RY(2*math.asin(1/math.sqrt(3)),wires=0)
        qml.PauliX(wires=0);qml.CRY(math.pi/2,wires=[0,1]);qml.PauliX(wires=0)
        for wire in range(5):qml.RY(math.pi/2,wires=wire+2)
        for layer in range(5):
            for scale,head in enumerate(model.base.heads):
                def branch():
                    for wire in range(5):qml.RY(inputs[scale][layer,wire],wires=wire+2)
                    f=head.filters[layer,0]
                    for a,b in head.pairs:
                        qml.RY(f[0],wires=a+2);qml.RY(f[1],wires=b+2)
                        qml.CRY(f[2],wires=[a+2,b+2]);qml.CRY(f[3],wires=[b+2,a+2])
                    for wire in range(7):qml.RY(head.weights[wire,layer],wires=wire+2)
                qml.ctrl(branch,control=[0,1],control_values=[scale//2,scale%2])()
            if layer<4:
                if dephase:
                    qml.PhaseDamping(1.,wires=0);qml.PhaseDamping(1.,wires=1)
                if model.variant=='local_capacity9':
                    angles=torch.cat((model.routing_angles[layer],model.phase_angles[layer]))
                    for wire in range(4):qml.RY(angles[wire],wires=wire+2)
                else:
                    if model.variant!='real_routing':
                        qml.RZ(model.phase_angles[layer,0],wires=1);qml.RZ(model.phase_angles[layer,1],wires=0)
                    qml.ctrl(qml.RY,control=[0,4],control_values=[0,1])(model.routing_angles[layer,0],wires=1)
                    qml.ctrl(qml.RY,control=[1,7],control_values=[0,1])(model.routing_angles[layer,1],wires=0)
        return qml.state()
    if not dephase:return circuit
    circuit.construct([], {});concrete=qml.transforms.convert_to_numpy_parameters(circuit.qtape)
    @qml.qnode(qml.device('default.mixed',wires=9),interface='torch',diff_method=None)
    def mixed():
        for op in concrete.operations:
            if isinstance(op,qml.ops.op_math.Controlled):qml.QubitUnitary(qml.matrix(op),wires=op.wires)
            else:qml.apply(op)
        return qml.probs(wires=range(9))
    return mixed


def primitives(model,data):
    """Batched, merged uniformly-controlled RY decomposition; includes preparation."""
    if model.variant=='classical_attention16':raise ValueError('Compile base heads')
    def arr(v):return v.detach().cpu().double().numpy() if isinstance(v,torch.Tensor) else np.asarray(v,dtype=np.float64)
    inputs=[arr(v) for v in data];batch=len(inputs[0]);ops=[]
    def ry(a,w):ops.append(('RY',(w,),np.asarray(a)))
    def mux(angles,controls,target):
        if any(np.ndim(a) for a in angles):table=np.stack([np.broadcast_to(a,(batch,)) for a in angles],-1)
        else:table=np.array(angles)[None]
        theta=compute_theta(table);code=gray_code(len(controls))
        for i in range(theta.shape[1]):
            angle=theta[:,i] if len(theta)>1 else theta[0,i]
            ry(angle,target);bit=(int(code[i],2)^int(code[(i+1)%len(code)],2)).bit_length()-1
            ops.append(('CNOT',(controls[::-1][bit],target),None))
    ry(2*math.asin(1/math.sqrt(3)),0);ops.append(('PauliX',(0,),None));mux([0,math.pi/2],[0],1);ops.append(('PauliX',(0,),None))
    for wire in range(5):ry(math.pi/2,wire+2)
    filters=[arr(h.filters) for h in model.base.heads];weights=[arr(h.weights) for h in model.base.heads]
    route=arr(model.routing_angles);phase=arr(model.phase_angles)
    for layer in range(5):
        for wire in range(5):mux([inputs[s][:,layer,wire] for s in range(3)]+[0.],[0,1],wire+2)
        for a,b in model.base.heads[0].pairs:
            for index,target in ((0,a),(1,b)):mux([filters[s][layer,0,index] for s in range(3)]+[0.],[0,1],target+2)
            for index,control,target in ((2,a,b),(3,b,a)):
                angles=np.zeros(8)
                for s in range(3):angles[2*s+1]=filters[s][layer,0,index]
                mux(angles,[0,1,control+2],target+2)
        for wire in range(7):mux([weights[s][wire,layer] for s in range(3)]+[0.],[0,1],wire+2)
        if layer<4:
            if model.variant=='local_capacity9':
                for wire,value in enumerate(np.concatenate((route[layer],phase[layer]))):ry(value,wire+2)
            else:
                if model.variant!='real_routing':ops.extend([('RZ',(1,),phase[layer,0]),('RZ',(0,),phase[layer,1])])
                mux([0,route[layer,0],0,0],[0,4],1);mux([0,route[layer,1],0,0],[1,7],0)
    return ops


@torch.no_grad()
def density(ops,batch,device='cpu',p1=0.,p2=0.,readout=0.):
    q=9;d=512;rho=torch.zeros(batch,d,d,dtype=torch.complex128,device=device);rho[:,0,0]=1.
    indices=torch.arange(d,device=device)
    for name,wires,angle in ops:
        if name=='RY':
            theta=torch.as_tensor(np.array(angle,copy=True),device=device,dtype=torch.float64)
            if theta.ndim==0:theta=theta.expand(batch)
            c=(theta/2).cos().reshape(batch,*([1]*(2*q-1)));s=(theta/2).sin().reshape(batch,*([1]*(2*q-1)))
            state=rho.reshape(batch,*([2]*(2*q)))
            for axis in (wires[0]+1,wires[0]+q+1):
                a,b=state.unbind(axis);state=torch.stack((c*a-s*b,s*a+c*b),axis)
            rho=depolarize(state.reshape(batch,d,d),wires,p1,q)
        elif name=='RZ':
            theta=torch.as_tensor(np.array(angle,copy=True),device=device,dtype=torch.float64)
            if theta.ndim==0:theta=theta.expand(batch)
            z=1.-2.*((indices>>(q-1-wires[0]))&1)
            phase=torch.exp(-.5j*theta[:,None]*z[None])
            rho=depolarize(phase[:,:,None]*rho*phase.conj()[:,None,:],wires,p1,q)
        elif name in ('CNOT','PauliX'):
            if name=='PauliX':perm=indices^(1<<(q-1-wires[0]))
            else:
                a,b=wires;perm=indices^(((indices>>(q-1-a))&1)<<(q-1-b))
            rho=depolarize(rho[:,perm][:,:,perm],wires,p2 if name=='CNOT' else p1,q)
        else:raise ValueError(name)
    diagonal=rho.diagonal(dim1=1,dim2=2)
    assert diagonal.imag.abs().max()<1e-9
    p=diagonal.real.clone();assert (p.sum(-1)-1).abs().max()<2e-9 and p.min()>-1e-9
    for wire in range(2,7):p=(1-readout)*p+readout*p[:,indices^(1<<(q-1-wire))]
    p=p.clamp_min(0);return (p/p.sum(-1,keepdim=True)).cpu().numpy()

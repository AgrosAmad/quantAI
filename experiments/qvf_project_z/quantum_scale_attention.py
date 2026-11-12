"""Prototype coherent communication among three scale-conditioned QVF branches.

Two address qubits label three scales; a shared seven-qubit payload undergoes
the corresponding branch's encoding and learned operations. Content-controlled
address mixers and relative phases are interleaved BEFORE subsequent branch
processing. No intermediate measurement or classical attention is used.

This is a nine-qubit direct-sum/multiplexed architecture, not three independent
seven-qubit registers, not a 21-qubit joint state, and not QK-softmax attention.
Older Z6 sources/checkpoints remain unchanged.
"""
import math
from collections import Counter
import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta,gray_code
import torch
from torch import nn

from .hires_models import build_model,component_seed,probabilities,resources as base_resources
from experiments.qvf_project_a.token_qvf import batched_ry


class CoherentScaleField(nn.Module):
    def __init__(self,seed=0,**kwargs):
        super().__init__()
        self.base=build_model('separate_scales',seed,**kwargs)
        self.cfg=self.base.cfg
        self.routing_angles=nn.Parameter(torch.empty(4,2))
        self.phase_angles=nn.Parameter(torch.empty(4,2))
        for name,parameter in [('routing_angles',self.routing_angles),('phase_angles',self.phase_angles)]:
            generator=torch.Generator().manual_seed(component_seed(seed,'coherent.'+name))
            with torch.no_grad():parameter.copy_(torch.randn(parameter.shape,generator=generator)*.15)
        basis=torch.arange(128)
        # Routing responds to encoded centre content and to a processed work wire.
        self.register_buffer('routing_masks',torch.stack((((basis>>4)&1).bool(),((basis>>1)&1).bool())),persistent=False)

    def encoded(self,codes,coordinates):return self.base.encoded(codes,coordinates)

    def initial_state(self,inputs):
        ref=inputs[0];dtype=torch.complex128 if ref.dtype==torch.float64 else torch.complex64
        payload=torch.zeros(len(ref),128,device=ref.device,dtype=dtype)
        payload[:,::4]=1/math.sqrt(32)  # Five |+> data wires, two |0> work wires.
        zero=torch.zeros_like(payload)
        return torch.stack((payload,payload,payload,zero),1)/math.sqrt(3)

    def mix(self,state,layer,routing=None,phases=None):
        routing=self.routing_angles if routing is None else routing
        phases=self.phase_angles if phases is None else phases
        # Exactly RZ on address wire1 followed by RZ on address wire0.
        signs=state.real.new_tensor([[1,1],[-1,1],[1,-1],[-1,-1]])
        phase=torch.exp(-.5j*(signs*phases[layer]).sum(-1))
        state=state*phase[None,:,None]
        for i,b in enumerate((1,2)):
            theta=routing[layer,i];mask=self.routing_masks[i]
            c=torch.where(mask,(theta/2).cos(),torch.ones_like(theta))
            s=torch.where(mask,(theta/2).sin(),torch.zeros_like(theta))
            pieces=list(state.unbind(1));a_value=pieces[0];b_value=pieces[b]
            pieces[0]=c*a_value-s*b_value;pieces[b]=s*a_value+c*b_value
            state=torch.stack(pieces,1)
        return state

    def quantum_state(self,inputs,routing=None,phases=None,trace_layers=False):
        if len(inputs)!=3 or any(x.shape[1:]!=(5,5) for x in inputs):
            raise ValueError('Three sets of five uploads to five data wires are required')
        state=self.initial_state(inputs);blocks=[h.blocks() for h in self.base.heads];history=[]
        for layer in range(5):
            pieces=[]
            for scale in range(3):
                payload=state[:,scale].reshape(-1,*([2]*7))
                for wire in range(5):payload=batched_ry(payload,inputs[scale][:,layer,wire],wire)
                pieces.append(payload.flatten(1)@blocks[scale][layer].to(state.dtype))
            state=torch.stack((*pieces,state[:,3]),1)
            if layer<4:state=self.mix(state,layer,routing,phases)
            if trace_layers:history.append(state)
        return (state.flatten(1),history) if trace_layers else state.flatten(1)

    def rgb_from_probabilities(self,p):
        # Trace out the scale address; read only the original five payload wires.
        marginal=p.reshape(-1,4,128).sum(1)
        z=marginal@self.base.heads[0].signs[:,:5].to(marginal.dtype)
        return torch.stack((z[:,0]+z[:,3],z[:,1]+z[:,4],z[:,2]),-1)

    def trace(self,codes,coordinates):
        inputs=self.encoded(codes,coordinates);state=self.quantum_state(inputs)
        p=probabilities(state);rgb=self.rgb_from_probabilities(p)
        return inputs,state,p,rgb.reshape(len(codes),len(coordinates),3)

    def forward(self,codes,coordinates):return self.trace(codes,coordinates)[-1]
    def cache_quantum(self):self.base.cache_quantum()
    def clear_cache(self):self.base.clear_cache()


def native_qnode(model,data,return_type='state',dephase=False):
    """Independent native controlled gates; default.mixed for coherence removal.

    Dephase both scale-label qubits before every interleaved mixer. All model
    parameters, branch gates, phases and readout remain identical. This is a
    small diagnostic, not an efficient full-batch mixed-state training engine.
    """
    if model.routing_angles.device.type!='cpu':raise ValueError('Use CPU/double for native audit')
    inputs=[torch.as_tensor(x,dtype=torch.float64) for x in data]
    device=qml.device('default.mixed' if dephase else 'default.qubit',wires=9)
    @qml.qnode(device,interface='torch',diff_method='backprop')
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
                qml.RZ(model.phase_angles[layer,0],wires=1)
                qml.RZ(model.phase_angles[layer,1],wires=0)
                qml.ctrl(qml.RY,control=[0,4],control_values=[0,1])(model.routing_angles[layer,0],wires=1)
                qml.ctrl(qml.RY,control=[1,7],control_values=[0,1])(model.routing_angles[layer,1],wires=0)
        if return_type=='probabilities':return qml.probs(wires=range(9))
        return qml.state()
    if not dephase:return circuit
    # The installed PennyLane 0.34 mixed device does not natively accept nested
    # Controlled operations. Use their independent small local unitary matrices
    # for this non-differentiated coherence-removal diagnostic, with no upgrade.
    circuit.construct([], {})
    concrete=qml.transforms.convert_to_numpy_parameters(circuit.qtape)
    @qml.qnode(device,interface='torch',diff_method=None)
    def mixed_circuit():
        for operation in concrete.operations:
            if isinstance(operation,qml.ops.op_math.Controlled):
                qml.QubitUnitary(qml.matrix(operation),wires=operation.wires)
            else:qml.apply(operation)
        return qml.probs(wires=range(9))
    return mixed_circuit


def primitive_ops(model,data):
    """RY/RZ/X/CNOT compilation with merged uniformly controlled rotations.

    Branch operations on distinct address subspaces commute. Their matching
    gates can therefore be grouped into one rotation multiplexer per gate slot,
    avoiding three separate generic multi-control decompositions. Gray-code
    synthesis includes all encoding and preparation gates and no work ancillas.
    """
    def array(value):
        return value.detach().cpu().double().numpy() if isinstance(value,torch.Tensor) else np.asarray(value,dtype=np.float64)
    inputs=[array(x) for x in data];operations=[]
    def ry(angle,wire):operations.append(('RY',(wire,),float(angle)))
    def multiplexer(angles,controls,target):
        values=np.asarray(angles,dtype=np.float64)
        assert values.shape==(2**len(controls),)
        theta=compute_theta(values[None])[0];code=gray_code(len(controls))
        for i,angle in enumerate(theta):
            ry(angle,target)
            bit=(int(code[i],2)^int(code[(i+1)%len(code)],2)).bit_length()-1
            operations.append(('CNOT',(controls[::-1][bit],target),None))
    ry(2*math.asin(1/math.sqrt(3)),0)
    operations.append(('PauliX',(0,),None));multiplexer([0.,math.pi/2],[0],1)
    operations.append(('PauliX',(0,),None))
    for wire in range(5):ry(math.pi/2,wire+2)
    heads=list(model.base.heads)
    assert all(h.pairs==heads[0].pairs for h in heads)
    filters=[array(h.filters) for h in heads];weights=[array(h.weights) for h in heads]
    phases=array(model.phase_angles);routing=array(model.routing_angles)
    for layer in range(5):
        for wire in range(5):multiplexer([inputs[s][layer,wire] for s in range(3)]+[0.],[0,1],wire+2)
        for a,b in heads[0].pairs:
            for index,target in ((0,a),(1,b)):
                multiplexer([filters[s][layer,0,index] for s in range(3)]+[0.],[0,1],target+2)
            for index,control,target in ((2,a,b),(3,b,a)):
                angles=np.zeros(8)
                for s in range(3):angles[2*s+1]=filters[s][layer,0,index]
                multiplexer(angles,[0,1,control+2],target+2)
        for wire in range(7):multiplexer([weights[s][wire,layer] for s in range(3)]+[0.],[0,1],wire+2)
        if layer<4:
            operations.append(('RZ',(1,),float(phases[layer,0])))
            operations.append(('RZ',(0,),float(phases[layer,1])))
            multiplexer([0.,routing[layer,0],0.,0.],[0,4],1)
            multiplexer([0.,routing[layer,1],0.,0.],[1,7],0)
    return operations


def compiled_resources(model,data):
    operations=primitive_ops(model,data)
    basis={'RY','RZ','CNOT','PauliX'}
    counts=dict(Counter(op[0] for op in operations))
    unsupported=sorted(set(counts)-basis)
    available=[0]*9
    for _,wires,_ in operations:
        if not wires:continue
        end=max(available[w] for w in wires)+1
        for wire in wires:available[wire]=end
    result=base_resources(model.base)
    result.update({'architecture':'coherent_scale_routing_prototype',
        'total_joint_qubits':9,'scale_address_qubits':2,'payload_qubits':7,
        'coherent_scale_branches':3,'state_amplitudes_per_coordinate':512,
        'independent_three_register_amplitudes_if_fully_joint_21q':2**21,
        'extra_stored_quantum_parameters':model.routing_angles.numel()+model.phase_angles.numel(),
        'total_stored_quantum_parameters':sum(p.numel() for h in model.base.heads for p in h.parameters())+16,
        'total_shared_parameters':sum(p.numel() for p in model.parameters()),
        'logical_quantum_queries_per_coordinate':1,
        'primitive_counts_including_preparation':counts,'primitive_basis_unresolved':unsupported,
        'all_to_all_primitive_depth':max(available),'routing_to_hardware_topology_included':False,
        'compiler':'Merged controlled branch rotations; Gray-code RY multiplexers, no ancillas. No global gate cancellation/routing optimization.',
        'classical_attention':False,'query_key_softmax_attention':False,
        'intermediate_measurement':False,'three_branches_can_execute_as_independent_jobs':False,
        'quantum_communication':'Coherent content-controlled address routing between conditional branch blocks',
        'readout':'Original five-payload-wire RGB after tracing out the scale address; no classical RGB fusion network',
        'scope':'Prototype; no image-quality, hardware-speedup, or novelty result'})
    # Remove inherited independent-head execution counts; they do not describe this circuit.
    for key in ('qubits_per_circuit','independent_circuits_per_coordinate','sequential_physical_qubits_required',
                'simultaneous_physical_qubits_if_parallel','quantum_queries_per_full_image',
                'primitive_RY_per_coordinate','primitive_RZ_per_coordinate','primitive_CNOT_per_coordinate',
                'inter_scale_entanglement','stored_quantum_parameters','shared_parameters'):
        result.pop(key,None)
    result['quantum_queries_per_full_image']=model.cfg.image_size**2
    return result

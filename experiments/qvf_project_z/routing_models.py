"""Z7 matched extensions; all Z6 and coherent-prototype sources stay frozen."""
from collections import Counter
import math
import numpy as np
import torch
from torch import nn
from .quantum_scale_attention import CoherentScaleField,primitive_ops as coherent_primitives
from .hires_models import build_model as old_build,resources as old_resources,probabilities
from experiments.qvf_project_a.token_qvf import batched_ry

NAMES=('coherent_routing','real_routing','local_capacity9','classical_attention16')


class RoutingField(CoherentScaleField):
    def __init__(self,name,seed=0,**kwargs):
        if name not in NAMES[:3]:raise ValueError(name)
        super().__init__(seed,**kwargs);self.variant=name
        if name=='real_routing':
            with torch.no_grad():self.phase_angles.zero_()
            self.phase_angles.requires_grad_(False)

    def mix(self,state,layer,routing=None,phases=None):
        if self.variant!='local_capacity9':return super().mix(state,layer,routing,phases)
        # Same 16 added stored/trainable parameters, acting on payload only.
        # Each address branch remains independent: no cross-scale input interaction.
        angles=torch.cat((self.routing_angles[layer],self.phase_angles[layer]))
        shape=state.shape;payload=state.reshape(-1,*([2]*7))
        for wire in range(4):payload=batched_ry(payload,angles[wire].expand(payload.shape[0]),wire)
        return payload.reshape(shape)


class AttentionFusionField(nn.Module):
    """16-parameter classical, coordinate/content-dependent softmax fusion.

    The gate sees seven same-bitstring statistics per branch, location, and scale.
    It can weight three quantum RGB outputs; it cannot directly predict RGB.
    This is output attention, not quantum feedback or attention over grid cells.
    """
    def __init__(self,seed=0,**kwargs):
        super().__init__();self.variant='classical_attention16'
        source=CoherentScaleField(seed,**kwargs);self.base=source.base;self.cfg=self.base.cfg
        init=torch.cat((source.routing_angles.detach().flatten(),source.phase_angles.detach().flatten()))
        self.content=nn.Parameter(init[:7].clone())
        self.position=nn.Parameter(init[7:13].reshape(3,2).clone())
        self.bias=nn.Parameter(init[13:].clone())

    def fuse(self,probs,coordinates,batch):
        signs=self.base.heads[0].signs.to(probs[0].dtype)
        features=[];rgbs=[]
        for p in probs:
            z=p@signs[:,:5]
            zz=p@(signs[:,2,None]*signs[:,[0,4]])
            features.append(torch.cat((z,zz),-1))
            rgbs.append(torch.stack((z[:,0]+z[:,3],z[:,1]+z[:,4],z[:,2]),-1))
        features=torch.stack(features,1);rgb=torch.stack(rgbs,1)
        pos=coordinates[None].expand(batch,-1,-1).reshape(-1,2)
        logits=features@self.content+pos@self.position.T+self.bias
        weights=logits.softmax(-1)
        return (weights[:,:,None]*rgb).sum(1),weights

    def trace(self,codes,coordinates):
        inputs=self.base.encoded(codes,coordinates)
        ps=[probabilities(h.state(x)) for h,x in zip(self.base.heads,inputs)]
        rgb,weights=self.fuse(ps,coordinates,len(codes))
        return inputs,ps,weights,rgb.reshape(len(codes),len(coordinates),3)

    def forward(self,codes,coordinates):return self.trace(codes,coordinates)[-1]
    def cache_quantum(self):self.base.cache_quantum()
    def clear_cache(self):self.base.clear_cache()


def build_model(name,seed=0,**kwargs):
    if name=='classical_attention16':return AttentionFusionField(seed,**kwargs)
    return RoutingField(name,seed,**kwargs)


def primitive_ops(model,data):
    if model.variant=='classical_attention16':raise ValueError('Compile the three base heads separately')
    source=coherent_primitives(model,data)
    if model.variant=='coherent_routing':return source
    output=[];i=0;stage=0
    while i<len(source):
        name,wires,angle=source[i]
        if name!='RZ':output.append(source[i]);i+=1;continue
        # A boundary consists of two RZ gates and two four-RY/four-CNOT multiplexers.
        assert source[i+1][0]=='RZ'
        if model.variant=='real_routing':
            output.extend(source[i+2:i+18])
        elif model.variant=='local_capacity9':
            values=torch.cat((model.routing_angles[stage],model.phase_angles[stage])).detach().cpu().double().numpy()
            output.extend(('RY',(wire+2,),float(values[wire])) for wire in range(4))
        i+=18;stage+=1
    assert stage==4
    return output


def resource_summary(model):
    classical=old_resources(model.base)['classical_parameters']
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion=model.variant=='classical_attention16'
    if fusion:
        old=old_resources(model.base)
        counts={'RY':old['primitive_RY_per_coordinate'],'CNOT':old['primitive_CNOT_per_coordinate']}
        q=7;calls=3;depth=None
    else:
        ops=primitive_ops(model,[torch.zeros(5,5) for _ in range(3)])
        counts=dict(Counter(x[0] for x in ops));q=9;calls=1;available=[0]*9
        for _,wires,_ in ops:
            end=max(available[w] for w in wires)+1
            for wire in wires:available[wire]=end
        depth=max(available)
    return {'variant':model.variant,'stored_shared_parameters':total,'trainable_shared_parameters':trainable,
        'classical_parameters':classical+(16 if fusion else 0),'stored_quantum_parameters':165 if fusion else 181,
        'trainable_quantum_parameters':165 if fusion else 173 if model.variant=='real_routing' else 181,
        'image_code_entries':model.cfg.latent_dim,'logical_circuits_per_coordinate':calls,
        'joint_qubits_per_circuit':q,'physical_qubits_sequential':q,'physical_qubits_simultaneous':21 if fusion else 9,
        'primitive_gates_per_coordinate':counts,'all_to_all_primitive_depth':depth,
        'quantum_calls_per_full_image':calls*model.cfg.image_size**2,
        'routing_to_device_topology_included':False,'measurement_bases':1,
        'classical_attention':fusion,'intermediate_measurements':False,
        'cross_scale_quantum_processing':model.variant in ('coherent_routing','real_routing'),
        'readout':'Classical softmax weighted quantum RGB' if fusion else 'Fixed original payload RGB after marginalizing scale address',
        'limits':['Parameter counts are stored/trainable entries, not a count of independent function degrees of freedom.',
            'Logical circuit calls do not measure primitive gate count or hardware latency.',
            'Same optimizer/coordinate budget does not match wall time, total shots, or total gates.']}


def readout_numpy(model,probs,coordinates,batch):
    from .hires_gates import marginal,SIGNS,RGB
    if model.variant!='classical_attention16':
        p=np.asarray(probs[0],dtype=np.float64).reshape(-1,4,128).sum(1)
        return marginal(p)@RGB
    ps=[marginal(p) for p in probs]
    features=[np.concatenate((p@SIGNS,p@(SIGNS[:,2,None]*SIGNS[:,[0,4]])),1) for p in ps]
    features=np.stack(features,1);rgb=np.stack([p@RGB for p in ps],1)
    pos=np.tile(np.asarray(coordinates,dtype=np.float64),(batch,1))
    content=model.content.detach().cpu().double().numpy();position=model.position.detach().cpu().double().numpy();bias=model.bias.detach().cpu().double().numpy()
    logits=features@content+pos@position.T+bias
    weights=np.exp(logits-logits.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
    return (weights[:,:,None]*rgb).sum(1)

"""Z6: four quantum changes with matched controls on genuine 256px targets.

All older model sources remain frozen. Quantum registers are independently
simulated; three heads are never treated as one entangled 21-qubit state.
"""
from dataclasses import dataclass, asdict
import hashlib
from types import SimpleNamespace

import torch
from torch import nn

from experiments.qvf_repro.qvf_fast import MLPEmbedding, positional_embedding, _apply_ry, _apply_cry
from experiments.qvf_project_a.token_qvf import batched_ry

NAMES = ('spatial_reference', 'joint_statistics', 'sequential_scales',
         'complex_phases', 'separate_scales', 'marginal_products',
         'extra_real_gates', 'three_mixed', 'attention_chain5')
PRIMARY = ('joint_statistics', 'sequential_scales', 'complex_phases', 'separate_scales')
SCHEDULE = (0, 0, 1, 1, 2)
OFFSETS = ((0.,-1.),(-1.,0.),(0.,0.),(1.,0.),(0.,1.))


@dataclass(frozen=True)
class HiresConfig:
    architecture: str = 'spatial_reference'
    image_size: int = 256
    grid_sizes: tuple = (16,32,64)
    channels: int = 4
    frequencies: int = 9
    width: int = 128

    @property
    def latent_dim(self):
        return self.channels*sum(s*s for s in self.grid_sizes)

    def validate(self):
        if self.architecture not in NAMES or len(self.grid_sizes)!=3:
            raise ValueError('Unknown architecture or scales')
        if self.channels!=4 or min(self.grid_sizes)<2 or self.frequencies<1 or self.image_size<2:
            raise ValueError('Invalid field configuration')


def sparse_bilinear(grid, coordinates):
    """Four-corner bilinear gather with border extension and align_corners=True.

    Avoids an O(number-of-queries * grid-area) interpolation weight matrix.
    Code-gradient accumulation uses PyTorch's deterministic index_select path.
    """
    b,c,h,w=grid.shape
    y=coordinates[:,0].clamp(0,1)*(h-1)
    x=coordinates[:,1].clamp(0,1)*(w-1)
    y0=y.floor().long();x0=x.floor().long()
    y1=(y0+1).clamp_max(h-1);x1=(x0+1).clamp_max(w-1)
    dy=y-y0;dx=x-x0
    flat=grid.flatten(2)
    parts=[flat.index_select(2, yy*w+xx)*weight[None,None]
        for yy,xx,weight in ((y0,x0,(1-dy)*(1-dx)),(y0,x1,(1-dy)*dx),
                             (y1,x0,dy*(1-dx)),(y1,x1,dy*dx))]
    return sum(parts).transpose(1,2)


class QuantumHead(nn.Module):
    def __init__(self, qubits=7, topology='star', extra='none'):
        super().__init__()
        self.n_qubits=qubits;self.topology=topology;self.extra=extra
        self.pairs=tuple((2,i) for i in range(qubits) if i!=2) if topology=='star' else \
            tuple((i,i+1) for parity in (0,1) for i in range(parity,qubits-1,2))
        self.weights=nn.Parameter(torch.empty(qubits,5))
        self.filters=nn.Parameter(torch.empty(5,1,4))
        if extra!='none':
            self.extra_single=nn.Parameter(torch.empty(5,5))
            self.extra_pair=nn.Parameter(torch.empty(5))
        basis=torch.arange(2**qubits)[:,None]
        signs=1.-2.*((basis>>torch.arange(qubits-1,-1,-1))&1).double()
        self.register_buffer('signs',signs,persistent=False)
        self._cache=None

    def block_matrix(self, layer):
        q=self.n_qubits;d=2**q
        complex_state=self.extra=='phase'
        state=torch.eye(d,dtype=torch.complex128 if complex_state else torch.float64).reshape(d,*([2]*q))
        w=self.weights.to('cpu',torch.float64);f=self.filters.to('cpu',torch.float64)[layer,0]
        signs=self.signs.to('cpu',torch.float64)
        if self.extra!='none':
            single=self.extra_single.to('cpu',torch.float64)[layer]
            pair=self.extra_pair.to('cpu',torch.float64)[layer]
            for wire in range(5):
                if complex_state:
                    phase=torch.exp(-.5j*single[wire]*signs[:,wire])
                    state=(state.reshape(d,d)*phase[None]).reshape(d,*([2]*q))
                else:
                    state=_apply_ry(state,single[wire],wire+1)
        for a,b in self.pairs:
            state=_apply_ry(state,f[0],a+1);state=_apply_ry(state,f[1],b+1)
            state=_apply_cry(state,f[2],a+1,b+1);state=_apply_cry(state,f[3],b+1,a+1)
            if self.extra=='phase':
                phase=torch.exp(-.5j*pair*signs[:,a]*signs[:,b])
                state=(state.reshape(d,d)*phase[None]).reshape(d,*([2]*q))
            elif self.extra=='real':
                state=_apply_cry(state,pair,a+1,b+1)
        for wire in range(q):
            state=_apply_ry(state,w[wire,layer],wire+1)
        dtype=torch.complex128 if self.weights.dtype==torch.float64 else torch.complex64
        return state.reshape(d,d).to(self.weights.device,dtype if complex_state else self.weights.dtype)

    def blocks(self):
        return self._cache if self._cache is not None else [self.block_matrix(i) for i in range(5)]

    def state(self, inputs, amplitude=False):
        blocks=self.blocks();q=self.n_qubits
        if amplitude:
            state=inputs.to(blocks[0].dtype)
            for block in blocks:state=state@block
            return state
        if inputs.shape[1:]!=(5,5):raise ValueError('Expected stages x five data wires')
        state=torch.zeros(len(inputs),2**q,device=inputs.device,dtype=blocks[0].dtype)
        state[:,0]=1.;state=state.reshape(-1,*([2]*q))
        for wire in range(5):
            state=batched_ry(state,inputs.new_full((len(inputs),),torch.pi/2),wire)
        for layer,block in enumerate(blocks):
            for wire in range(5):state=batched_ry(state,inputs[:,layer,wire],wire)
            state=(state.flatten(1)@block).reshape(-1,*([2]*q))
        return state.flatten(1)


def probabilities(state):
    return state.real.square()+state.imag.square() if state.is_complex() else state.square()


class HiresField(nn.Module):
    def __init__(self, config):
        super().__init__();config.validate();self.cfg=config
        name=config.architecture;self.is_attention=name=='attention_chain5'
        self.values=nn.ModuleList(nn.Linear(4,4) for _ in range(3))
        pos_width=2+4*config.frequencies
        self.register_buffer('offsets',torch.tensor(OFFSETS),persistent=False)
        masks=torch.zeros(3,12)
        for i in range(3):masks[i,i*4:(i+1)*4]=1.
        self.register_buffer('scale_masks',masks,persistent=False)
        if self.is_attention:
            self.queries=nn.ModuleList(nn.Linear(pos_width,16) for _ in range(3))
            self.keys=nn.ModuleList(nn.Linear(6,16) for _ in range(3))
            self.encoder=MLPEmbedding(12+pos_width,config.width,32,3)
            for i,size in enumerate(config.grid_sizes):
                y,x=torch.meshgrid(torch.linspace(0,1,size),torch.linspace(0,1,size),indexing='ij')
                self.register_buffer(f'positions_{i}',torch.stack((y,x),-1).reshape(-1,2),persistent=False)
        else:
            self.encoder=MLPEmbedding(12+pos_width+2,config.width,5,3,amplitude_output=False)
        count=3 if name in ('separate_scales','three_mixed') else 1
        extra='phase' if name=='complex_phases' else 'real' if name=='extra_real_gates' else 'none'
        self.heads=nn.ModuleList(QuantumHead(5 if self.is_attention else 7,
            'chain' if self.is_attention else 'star',extra) for _ in range(count))
        if name in ('joint_statistics','marginal_products'):
            self.readout=nn.Linear(9,3)

    def spatial_features(self, codes, coordinates):
        gathered=[];offset=0
        for i,size in enumerate(self.cfg.grid_sizes):
            count=4*size*size;grid=codes[:,offset:offset+count].reshape(-1,4,size,size)
            positions=coordinates[:,None]+self.offsets.to(codes.dtype)[None]/(size-1)
            raw=sparse_bilinear(grid,positions.flatten(0,1)).reshape(len(codes),len(coordinates),5,4)
            gathered.append(self.values[i](raw));offset+=count
        return torch.cat(gathered,-1)

    def attention_input(self, codes, coordinates, position):
        features=[];offset=0
        for i,size in enumerate(self.cfg.grid_sizes):
            count=4*size*size
            tokens=codes[:,offset:offset+count].reshape(-1,4,size*size).transpose(1,2)
            centres=getattr(self,f'positions_{i}').to(codes.dtype)
            keys=self.keys[i](torch.cat((tokens,centres[None].expand(len(codes),-1,-1)),-1))
            queries=self.queries[i](position)
            scores=torch.einsum('pd,bnd->bpn',queries,keys)/4.
            distance=((coordinates[:,None]-centres[None])*(size-1)).square().sum(-1)
            weights=(scores-2.*distance).softmax(-1)
            features.append(weights@self.values[i](tokens));offset+=count
        return self.encoder(torch.cat((*features,position[None].expand(len(codes),-1,-1)),-1).flatten(0,1))

    def encoded(self, codes, coordinates):
        if codes.shape[1]!=self.cfg.latent_dim:raise ValueError('Wrong code size')
        position=positional_embedding(coordinates,self.cfg.frequencies)
        if self.is_attention:return [self.attention_input(codes,coordinates,position)]
        features=self.spatial_features(codes,coordinates)
        pos=position[None,:,None].expand(len(codes),-1,5,-1)
        roles=self.offsets.to(codes.dtype)[None,None].expand(len(codes),len(coordinates),-1,-1)
        def encode(f):
            inputs=torch.cat((f,pos,roles),-1).flatten(0,2)
            angles=torch.pi*self.encoder(inputs).tanh()
            return angles.reshape(len(codes)*len(coordinates),5,5).transpose(-1,-2)
        if self.cfg.architecture in ('separate_scales','sequential_scales'):
            by_scale=[encode(features*self.scale_masks[i].to(codes.dtype)) for i in range(3)]
            if self.cfg.architecture=='separate_scales':return by_scale
            return [torch.stack([by_scale[scale][:,layer] for layer,scale in enumerate(SCHEDULE)],1)]
        encoded=encode(features)
        return [encoded]*len(self.heads)

    def read_probabilities(self, probs):
        zs=[p@head.signs[:,:5].to(p.dtype) for p,head in zip(probs,self.heads)]
        if hasattr(self,'readout'):
            z=zs[0]
            if self.cfg.architecture=='joint_statistics':
                signs=self.heads[0].signs.to(z.dtype)
                products=probs[0]@(signs[:,2,None]*signs[:,[0,1,3,4]])
            else:products=z[:,2,None]*z[:,[0,1,3,4]]
            return self.readout(torch.cat((z,products),-1))
        rgb=[torch.stack((z[:,0]+z[:,3],z[:,1]+z[:,4],z[:,2]),-1) for z in zs]
        return torch.stack(rgb).mean(0)

    def trace(self, codes, coordinates):
        inputs=self.encoded(codes,coordinates)
        probs=[probabilities(head.state(data,self.is_attention)) for head,data in zip(self.heads,inputs)]
        return inputs,probs,self.read_probabilities(probs).reshape(len(codes),len(coordinates),3)

    def forward(self, codes, coordinates):
        return self.trace(codes,coordinates)[2]

    @torch.no_grad()
    def cache_quantum(self):
        for head in self.heads:head._cache=[head.block_matrix(i) for i in range(5)]

    def clear_cache(self):
        for head in self.heads:head._cache=None


def component_seed(seed,name):
    return (seed+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little'))%(2**63-1)


def build_model(name,seed=0,**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=HiresField(HiresConfig(architecture=name,**kwargs))
        for module_name,module in model.named_modules():
            if isinstance(module,nn.Linear):
                torch.manual_seed(component_seed(seed,module_name));module.reset_parameters()
                if module_name=='encoder.layers.3' and not model.is_attention:
                    nn.init.normal_(module.weight,std=.01);nn.init.zeros_(module.bias)
        for i,head in enumerate(model.heads):
            for key,parameter in head.named_parameters():
                torch.manual_seed(component_seed(seed,f'heads.{i}.{key}'))
                nn.init.normal_(parameter,std=.03 if key.startswith('extra') else .1)
        if hasattr(model,'readout'):
            with torch.no_grad():
                model.readout.weight.zero_();model.readout.bias.zero_()
                model.readout.weight[0,[0,3]]=1.;model.readout.weight[1,[1,4]]=1.;model.readout.weight[2,2]=1.
    return model


def resources(model):
    q=model.heads[0].n_qubits;count=len(model.heads);pairs=len(model.heads[0].pairs)
    ry=5+25+5*(6*pairs+q);rz=0;cnot=20*pairs
    if model.is_attention:ry=2**q-1+5*(6*pairs+q);cnot+=2**q-2
    elif model.cfg.architecture=='complex_phases':rz=25+5*pairs;cnot+=10*pairs
    elif model.cfg.architecture=='extra_real_gates':ry+=25+10*pairs;cnot+=10*pairs
    quantum=sum(p.numel() for h in model.heads for p in h.parameters())
    total=sum(p.numel() for p in model.parameters())
    return {'shared_parameters':total,'classical_parameters':total-quantum,
        'stored_quantum_parameters':quantum,'structurally_redundant_terminal_work_rotations':(q-5)*count,
        'qubits_per_circuit':q,'independent_circuits_per_coordinate':count,
        'sequential_physical_qubits_required':q,'simultaneous_physical_qubits_if_parallel':q*count,
        'image_code_entries':model.cfg.latent_dim,'grid_sizes':list(model.cfg.grid_sizes),
        'quantum_queries_per_full_image':count*model.cfg.image_size**2,
        'primitive_RY_per_coordinate':ry*count,'primitive_RZ_per_coordinate':rz*count,
        'primitive_CNOT_per_coordinate':cnot*count,
        'classical_slot_mlp_evaluations_per_coordinate':0 if model.is_attention else 15 if model.cfg.architecture in ('separate_scales','sequential_scales') else 5,
        'measurement_bases':1,'joint_statistics_within_circuit_only':True,
        'inter_scale_entanglement':False if count==3 else 'Possible through sequential processing; not an entanglement certification',
        'readout': 'learned affine map of five singles plus four joint ZZ statistics' if model.cfg.architecture=='joint_statistics' else
            'learned affine map of five singles plus four classical products of marginal means' if model.cfg.architecture=='marginal_products' else
            'fixed average across heads of Z0+Z3, Z1+Z4, Z2'}

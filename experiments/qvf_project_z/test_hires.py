"""Independent numeric tests for Z6; no training data required."""
import itertools
import math
import unittest
import numpy as np
import torch

from experiments.qvf_project_a.conv_qvf import local_features
from experiments.qvf_project_a.train import configure_runtime
from .hires_models import NAMES, build_model, sparse_bilinear, probabilities, resources
from .hires_gates import (pennylane_state, primitives, density, readout_numpy,
                         expected_mse, gate_resources)


class HiresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime('cpu', 17)

    def model(self,name):
        return build_model(name,17,image_size=16,grid_sizes=(2,4,8),frequencies=3).double()

    def test_sparse_sampler(self):
        torch.manual_seed(5)
        grid=torch.randn(2,4,5,7,dtype=torch.float64,requires_grad=True)
        coords=torch.cat((torch.rand(23,2,dtype=torch.float64),torch.tensor([[0.,0.],[1.,1.],[-.2,1.3]],dtype=torch.float64)))
        a=sparse_bilinear(grid,coords);b=local_features(grid,coords)
        torch.testing.assert_close(a,b,atol=2e-14,rtol=2e-14)
        probe=torch.randn_like(a)
        ga=torch.autograd.grad((a*probe).sum(),grid)[0]
        gb=torch.autograd.grad((b*probe).sum(),grid)[0]
        torch.testing.assert_close(ga,gb,atol=2e-14,rtol=2e-14)

    def test_native_quantum_states_and_gradients(self):
        for name in ('spatial_reference','complex_phases','extra_real_gates','attention_chain5'):
            with self.subTest(name=name):
                model=self.model(name);head=model.heads[0]
                x=torch.randn(32 if model.is_attention else (5,5),dtype=torch.float64,requires_grad=True)
                data=x.softmax(-1).sqrt() if model.is_attention else x
                state=head.state(data[None],model.is_attention)[0]
                independent=pennylane_state(head,data,model.is_attention)
                torch.testing.assert_close(state.to(torch.complex128),independent,atol=3e-13,rtol=3e-13)
                obs=torch.arange(len(state),dtype=torch.float64).sin()
                params=[x,*head.parameters()]
                g1=torch.autograd.grad((probabilities(state)*obs).sum(),params,retain_graph=True)
                g2=torch.autograd.grad((probabilities(independent)*obs).sum(),params)
                for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-11)
                if name=='complex_phases':
                    self.assertGreater(g1[-1].abs().max().item(),1e-5)
                ops=primitives(head,data.detach()[None].numpy(),model.is_attention)
                p=density(ops,1,head.n_qubits)
                np.testing.assert_allclose(p[0],probabilities(independent).detach().numpy(),atol=3e-13)
                counts={k:sum(op[0]==k for op in ops) for k in ('RY','RZ','CNOT')}
                declared=resources(model)
                for k,v in counts.items():self.assertEqual(v,declared[f'primitive_{k}_per_coordinate'])

    def test_initialization_controls_and_all_fields(self):
        ref=self.model('spatial_reference');codes=torch.randn(2,ref.cfg.latent_dim,dtype=torch.float64)*.01
        coords=torch.rand(7,2,dtype=torch.float64)
        reference=ref(codes,coords)
        for name in NAMES:
            with self.subTest(name=name):
                model=self.model(name)
                _,p,y=model.trace(codes,coords)
                np.testing.assert_allclose(readout_numpy(model,[v.detach().numpy() for v in p]),y.detach().flatten(0,1).numpy(),atol=2e-13)
                y.square().mean().backward()
                self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters()))
                if name in ('joint_statistics','marginal_products'):
                    torch.testing.assert_close(y,reference,atol=0,rtol=0)
                if name!='attention_chain5':
                    for k,v in ref.encoder.state_dict().items():torch.testing.assert_close(v,model.encoder.state_dict()[k],atol=0,rtol=0)
        phase=self.model('complex_phases')
        with torch.no_grad():
            phase.heads[0].extra_single.zero_();phase.heads[0].extra_pair.zero_()
        torch.testing.assert_close(phase(codes,coords),reference,atol=3e-13,rtol=3e-13)
        a=self.model('separate_scales');b=self.model('three_mixed')
        for x,y in zip(a.heads.parameters(),b.heads.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
        self.assertFalse(torch.equal(a.heads[0].weights,a.heads[1].weights))

    def test_joint_information_is_distinct(self):
        p=np.zeros((1,32));p[0,[0,31]]=.5
        q=np.full((1,32),1/32)
        for name in ('joint_statistics','marginal_products'):
            m=self.model(name)
            with torch.no_grad():m.readout.weight.zero_();m.readout.weight[:,5:]=1.;m.readout.bias.zero_()
            difference=np.abs(readout_numpy(m,[p])-readout_numpy(m,[q])).max()
            self.assertEqual(difference,4. if name=='joint_statistics' else 0.)

    def test_exact_finite_shot_mse(self):
        target=np.array([[.2,-.4,.6]])
        p=np.zeros((1,32));p[0,2]=.3;p[0,27]=.7
        for name in ('spatial_reference','joint_statistics','marginal_products','separate_scales'):
            m=self.model(name)
            if hasattr(m,'readout'):
                with torch.no_grad():
                    m.readout.weight.copy_(torch.arange(27,dtype=torch.float64).reshape(3,9).sin()*.3)
                    m.readout.bias.copy_(torch.tensor([.1,-.1,.2]))
            heads=len(m.heads)
            for n in (1,2,3):
                exact=0.
                for counts in itertools.product(range(n+1),repeat=heads):
                    weight=math.prod(math.comb(n,k)*.3**k*.7**(n-k) for k in counts)
                    ps=[]
                    for k in counts:
                        sample=np.zeros_like(p);sample[0,2]=k/n;sample[0,27]=1-k/n;ps.append(sample)
                    exact+=weight*((readout_numpy(m,ps)-target)**2).mean()
                np.testing.assert_allclose(expected_mse(m,[p]*heads,target,n)[0],exact,rtol=3e-13,atol=3e-13)


if __name__=='__main__':unittest.main()

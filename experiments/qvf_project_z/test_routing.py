"""Checks for Z7 controls and independent gates before long training."""
from collections import Counter
import unittest
import numpy as np
import torch
from experiments.qvf_project_a.train import configure_runtime
from .hires_models import probabilities
from .routing_models import NAMES,build_model,readout_numpy,resource_summary,primitive_ops
from .routing_gates import native_qnode,primitives,density


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure_runtime('cpu',71)
    def model(self,name):return build_model(name,4,image_size=16,grid_sizes=(2,4,8),frequencies=3).double()

    def test_field_readouts_and_pairing(self):
        reference=self.model(NAMES[0]);codes=torch.randn(2,336,dtype=torch.float64)*.01;points=torch.rand(3,2,dtype=torch.float64)
        for name in NAMES:
            model=self.model(name)
            for key,value in reference.base.state_dict().items():torch.testing.assert_close(value,model.base.state_dict()[key],atol=0,rtol=0)
            result=model.trace(codes,points);ps=result[1] if name=='classical_attention16' else [result[2]]
            rgb=result[-1];independent=readout_numpy(model,[v.detach().numpy() for v in ps],points.numpy(),2)
            np.testing.assert_allclose(independent,rgb.detach().flatten(0,1).numpy(),atol=3e-12)
            rgb.square().mean().backward()
            for param in model.parameters():
                if param.requires_grad:self.assertTrue(torch.isfinite(param.grad).all())
            self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in reference.parameters()))
        fusion=self.model('classical_attention16')
        with torch.no_grad():fusion.content.zero_();fusion.position.zero_();fusion.bias.zero_()
        torch.testing.assert_close(fusion(codes,points),fusion.base(codes,points),atol=3e-12,rtol=3e-12)

    def test_native_states_gradients_and_compiler(self):
        for name in NAMES[:3]:
            with self.subTest(name=name):
                model=self.model(name);g=torch.Generator().manual_seed(24)
                data=[torch.randn(1,5,5,generator=g,dtype=torch.float64,requires_grad=True) for _ in range(3)]
                state=model.quantum_state(data)[0];native=native_qnode(model,[v[0] for v in data])()
                torch.testing.assert_close(state,native,atol=3e-12,rtol=3e-12)
                observable=torch.arange(512,dtype=torch.float64).sin()
                params=[*data,*[v for n,v in model.named_parameters() if v.requires_grad and (n.startswith(('routing','phase')) or n.startswith('base.heads'))]]
                ga=torch.autograd.grad((probabilities(state)*observable).sum(),params,retain_graph=True)
                gb=torch.autograd.grad((probabilities(native)*observable).sum(),params)
                for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=3e-11,rtol=3e-10)
                ops=primitives(model,data);single=primitive_ops(model,[v[0] for v in data])
                self.assertEqual(Counter(x[0] for x in ops),Counter(x[0] for x in single))
                for a,b in zip(ops,single):
                    self.assertEqual(a[:2],b[:2])
                    if a[2] is not None:np.testing.assert_allclose(a[2],b[2],atol=1e-14)
                # Small independent density execution includes every compiled gate.
                p=density(ops,1)
                np.testing.assert_allclose(p[0],probabilities(native).detach().numpy(),atol=4e-11,rtol=4e-10)

    def test_local_capacity_has_no_cross_scale_interaction(self):
        model=self.model('local_capacity9')
        x=[torch.randn(1,5,5,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        red=model.rgb_from_probabilities(probabilities(model.quantum_state(x)))[0,0]
        coarse=torch.autograd.grad(red,x[0],create_graph=True)[0]
        cross=torch.autograd.grad(coarse.sum(),x[1])[0]
        self.assertLess(float(cross.abs().max()),1e-12)
        self.assertEqual(resource_summary(model)['stored_quantum_parameters'],181)


if __name__=='__main__':unittest.main()

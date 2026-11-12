"""Original QVF equivalence, quantum compilation and the full registry."""
import unittest
import numpy as np
import torch
from .coverage_models import NAMES,build_model,vanilla_native,vanilla_primitives,resources
from .hires_gates import density
from .hires_models import probabilities


class CoverageTests(unittest.TestCase):
    def test_vanilla_exact_field_cache_and_gradients(self):
        torch.set_num_threads(4)
        for frequencies in (6,9):
            model=build_model('vanilla_qvf',0,image_size=32,grid_sizes=(2,4,8),channels=4,frequencies=frequencies,width=128).double()
            codes=torch.randn(2,64,dtype=torch.float64,requires_grad=True)*.01
            xy=torch.rand(7,2,dtype=torch.float64,requires_grad=True)
            actual=model(codes,xy)
            reference=model.field(model.inputs(codes,xy),per_sample=True).reshape(2,7,3)
            self.assertLess(float((actual-reference).abs().max()),1e-12)
            params=[codes,xy,*model.parameters()]
            aa=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
            bb=torch.autograd.grad(reference.square().sum(),params)
            self.assertLess(max(float((a-b).abs().max()) for a,b in zip(aa,bb)),1e-11)
            model.cache_quantum()
            self.assertLess(float((model(codes,xy)-actual).abs().max()),1e-12)
            model.clear_cache()

    def test_original_quantum_native_and_compiled(self):
        model=build_model('vanilla_qvf',1,image_size=32,frequencies=6,width=128).double()
        x=torch.randn(32,dtype=torch.float64).softmax(-1).sqrt().requires_grad_()
        xn=x/x.norm();fast=xn@model.field.quantum.circuit_matrix();native=vanilla_native(model,xn)
        self.assertLess(float((fast-native).abs().max()),1e-12)
        obs=torch.arange(32,dtype=torch.float64).sin();params=[x,*model.field.quantum.parameters()]
        aa=torch.autograd.grad((fast.square()*obs).sum(),params,retain_graph=True)
        bb=torch.autograd.grad((probabilities(native)*obs).sum(),params)
        self.assertLess(max(float((a-b).abs().max()) for a,b in zip(aa,bb)),1e-11)
        ops=vanilla_primitives(model,xn.detach().numpy()[None])
        result=density(ops,1,5)
        self.assertLess(float(np.max(np.abs(result-fast.detach().square().numpy()[None]))),1e-12)
        info=resources(model,'vanilla_qvf')
        self.assertEqual(info['stored_quantum_parameters'],150)
        self.assertEqual(info['primitive_gates_per_coordinate'],{'RY':256,'CNOT':230})

    def test_all_methods_forward_backward(self):
        for name in NAMES:
            model=build_model(name,0,image_size=32,grid_sizes=(2,4,8),channels=4,frequencies=6,width=128)
            codes=(torch.randn(2,model.cfg.latent_dim)*.01).requires_grad_()
            y=model(codes,torch.rand(5,2));self.assertEqual(tuple(y.shape),(2,5,3))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(codes.grad).all(),name)
            for p in model.parameters():
                if p.requires_grad:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),name)


if __name__=='__main__':unittest.main(verbosity=2)

"""Physical and differentiability checks for the coherent scale-routing prototype."""
import unittest
import numpy as np
import torch
import pennylane as qml
from experiments.qvf_project_a.train import configure_runtime
from .hires_models import probabilities
from .quantum_scale_attention import CoherentScaleField,native_qnode,compiled_resources,primitive_ops
from .hires_gates import sample_probabilities,RGB
from .spatial_gates import shot_mse_moments


class CoherentScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure_runtime('cpu',23)

    def model(self):
        return CoherentScaleField(23,image_size=16,grid_sizes=(2,4,8),frequencies=3).double()

    def inputs(self):
        g=torch.Generator().manual_seed(77)
        return [torch.randn(1,5,5,generator=g,dtype=torch.float64,requires_grad=True) for _ in range(3)]

    def test_native_state_and_all_quantum_gradients(self):
        model=self.model();data=self.inputs()
        fast=model.quantum_state(data)[0]
        native=native_qnode(model,[x[0] for x in data])()
        torch.testing.assert_close(fast,native,atol=3e-12,rtol=3e-12)
        observable=torch.arange(512,dtype=torch.float64).sin()
        params=[*data,model.routing_angles,model.phase_angles,*model.base.heads.parameters()]
        ga=torch.autograd.grad((probabilities(fast)*observable).sum(),params,retain_graph=True)
        gb=torch.autograd.grad((probabilities(native)*observable).sum(),params)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=3e-11,rtol=3e-10)
        for gradient in ga[:5]:self.assertGreater(float(gradient.abs().max()),1e-8)

    def test_disabled_routing_is_original_three_head_mean(self):
        model=self.model();codes=torch.randn(2,model.cfg.latent_dim,dtype=torch.float64)*.01
        coordinates=torch.rand(5,2,dtype=torch.float64);inputs=model.encoded(codes,coordinates)
        state=model.quantum_state(inputs,routing=torch.zeros_like(model.routing_angles))
        prediction=model.rgb_from_probabilities(probabilities(state)).reshape(2,5,3)
        torch.testing.assert_close(prediction,model.base(codes,coordinates),atol=3e-12,rtol=3e-12)
        # Relative phases alone cannot change this address-traced readout.
        alternate=model.quantum_state(inputs,routing=torch.zeros_like(model.routing_angles),phases=torch.ones_like(model.phase_angles)*1.7)
        torch.testing.assert_close(probabilities(state),probabilities(alternate),atol=3e-12,rtol=3e-12)

    def test_norm_empty_address_and_final_mixer_pitfall(self):
        model=self.model();data=self.inputs();state,history=model.quantum_state(data,trace_layers=True)
        for intermediate in history:
            torch.testing.assert_close(probabilities(intermediate).sum((1,2)),torch.ones(1,dtype=torch.float64),atol=3e-12,rtol=3e-12)
            self.assertEqual(float(intermediate[:,3].abs().max()),0.)
        # An address-only unitary after all processing cannot change payload RGB.
        generator=torch.Generator().manual_seed(88)
        unitary=torch.linalg.qr(torch.randn(4,4,generator=generator,dtype=torch.float64))[0].to(torch.complex128)
        mixed=torch.einsum('st,btd->bsd',unitary,state.reshape(-1,4,128)).flatten(1)
        torch.testing.assert_close(model.rgb_from_probabilities(probabilities(state)),model.rgb_from_probabilities(probabilities(mixed)),atol=3e-12,rtol=3e-12)

    def test_coherence_and_phase_have_observable_effect(self):
        model=self.model();data=self.inputs()
        with torch.no_grad():
            model.routing_angles.copy_(torch.tensor([[.4,-.7],[.6,.2],[-.3,.5],[.8,-.4]]))
            model.phase_angles.copy_(torch.tensor([[.3,.8],[-.7,.2],[.4,-.6],[.2,.7]]))
        coherent=probabilities(model.quantum_state(data))
        without_phases=probabilities(model.quantum_state(data,phases=torch.zeros_like(model.phase_angles)))
        phase_effect=float((model.rgb_from_probabilities(coherent)-model.rgb_from_probabilities(without_phases)).abs().max())
        with torch.no_grad():dephased=torch.as_tensor(native_qnode(model,[x[0] for x in data],return_type='probabilities',dephase=True)())[None]
        coherence_effect=float((model.rgb_from_probabilities(coherent)-model.rgb_from_probabilities(dephased)).abs().max())
        self.assertGreater(phase_effect,1e-5);self.assertGreater(coherence_effect,1e-5)
        print(f'phase RGB effect={phase_effect:.6g}; dephasing RGB effect={coherence_effect:.6g}',flush=True)

    def test_compiled_primitive_state(self):
        model=self.model();data=[x[0] for x in self.inputs()]
        operations=primitive_ops(model,data)
        self.assertTrue(all(op[0] in {'RY','RZ','CNOT','PauliX'} for op in operations))
        @qml.qnode(qml.device('default.qubit',wires=9))
        def primitive():
            for name,wires,angle in operations:
                if angle is None:getattr(qml,name)(wires=wires)
                else:getattr(qml,name)(angle,wires=wires)
            return qml.probs(wires=range(9))
        independent=primitive()
        reference=probabilities(model.quantum_state([x[None] for x in data])[0]).detach().numpy()
        np.testing.assert_allclose(independent,reference,atol=4e-11,rtol=4e-10)
        counts=compiled_resources(model,data)
        self.assertFalse(counts['primitive_basis_unresolved'])
        self.assertEqual(counts['total_joint_qubits'],9)
        self.assertGreater(counts['primitive_counts_including_preparation']['CNOT'],360)

    def test_finite_shots_use_one_joint_circuit_distribution(self):
        model=self.model();data=self.inputs()
        with torch.no_grad():p=probabilities(model.quantum_state(data)).reshape(1,4,128).sum(1).numpy()
        p=p/p.sum(-1,keepdims=True);target=np.array([[.2,.5,.8]])
        expectation,variance=shot_mse_moments(p,target,256)
        trials=[]
        for seed in range(400,600):
            rgb=sample_probabilities(p,256,seed)@RGB
            trials.append(float(((rgb-target)**2).mean()))
        error=abs(np.mean(trials)-expectation[0]);sem=np.sqrt(variance[0]/len(trials))
        self.assertLess(error,6*sem)

    def test_complete_field_backprop_and_checkpoint(self):
        model=self.model();codes=(torch.randn(2,model.cfg.latent_dim,dtype=torch.float64)*.01).requires_grad_()
        coordinates=torch.rand(4,2,dtype=torch.float64);output=model(codes,coordinates)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(codes.grad).all());self.assertGreater(float(codes.grad.abs().max()),0.)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
        clone=self.model();clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone(codes,coordinates),output,atol=0,rtol=0)


if __name__=='__main__':unittest.main()

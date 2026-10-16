"""Common pipeline correctness; timings are checked for schema, never for speed."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from mlcompiler import (CPUBackend, Diagnostic, Graph, Node, TensorSpec, analysis,
                        compare_outputs, compile_graph, from_legacy, infer_shapes,
                        native, onnx, pytorch, validate)
from mlcompiler.experiment import run_suite, save
from mlcompiler.measurement import MeasureConfig
from mlcompiler.passes import (AffineReLUFusion, AlgebraicSimplification, ConstantFolding,
    DeadCodeElimination, LayoutSimplification, PassManager, PassResult)
from mlcompiler.workloads import NAMES, build


class IRTests(unittest.TestCase):
    def test_invalid_graphs(self):
        x = Node('x', 'input', spec=TensorSpec((2,), torch.float32))
        for graph in (Graph((x, x), ('x',)), Graph((x,), ('missing',)),
                      Graph((Node('y', 'relu', ('x',)), x), ('y',)),
                      Graph((x, Node('y', 'bad', ('x',))), ('y',)),
                      Graph((x, Node('y', 'add', ('x',))), ('y',)),
                      Graph((replace(x, spec=TensorSpec((-1,), torch.float32)),), ('x',)),
                      Graph((x, Node('y', 'relu', ('x',), attrs={'inplace': True})), ('y',))):
            with self.subTest(graph=repr(graph)), self.assertRaises(Diagnostic): graph.validate()

    def test_topology_output_alias_and_dynamic_shape(self):
        g = Graph((Node('x', 'input', spec=TensorSpec((None, 2), torch.float32)),), ('x','x'))
        g.validate()
        plan = CPUBackend().compile(g)
        self.assertEqual(len(plan(torch.ones(3,2))), 2)
        with self.assertRaises(Diagnostic): plan(torch.ones(3,4))
        with self.assertRaises(Diagnostic): plan(torch.ones(3,2).double())
        self.assertEqual(g.topological(), g.ops)

    def test_static_shape_mismatch_and_metadata(self):
        g = Graph((Node('x','input', spec=TensorSpec((2,3),torch.float32)),
                   Node('w','input', spec=TensorSpec((4,5),torch.float32)),
                   Node('y','matmul',('x','w'))),('y',))
        with self.assertRaises(Diagnostic): infer_shapes(g)
        w = build('affine',8)
        stats = analysis(w.graph)
        self.assertEqual(stats['compute_ops'],3)
        self.assertEqual(stats['estimated_compute_output_bytes'],8*4*3)
        self.assertIn('not measured',stats['memory_note'])

    def test_unknown_input_shape_checks_known_output_at_runtime(self):
        graph = Graph((Node('x', 'input', spec=TensorSpec((None, 2), torch.float32)),
                       Node('y', 'relu', ('x',), TensorSpec((3, 2), torch.float32))), ('y',))
        plan = CPUBackend().compile(graph)
        with self.assertRaises(Diagnostic): plan(torch.ones(4, 2))
        self.assertEqual(tuple(plan(torch.ones(3, 2))[0].shape), (3, 2))

    def test_legacy_adapter(self):
        from compiler import AffineReLU, capture
        args = (torch.tensor([-1.,2.]),torch.tensor(1.5),torch.tensor(.5))
        adapted = from_legacy(capture(AffineReLU(),args))
        compare_outputs((AffineReLU()(*args),),CPUBackend().compile(adapted)(*args))


class PassTests(unittest.TestCase):
    def run_pass(self, name, optimization):
        w=build(name,8)
        result=PassManager((optimization,)).run(w.graph,w.samples)
        self.assertTrue(any(r.status=='applied' and r.validated for r in result.records))
        self.assertTrue(validate(w.graph,result.graph,(w.evaluation_inputs,)).passed)
        self.assertTrue(result.converged)
        return w,result

    def test_constant_folding(self):
        w,result=self.run_pass('constant',ConstantFolding())
        self.assertTrue(any(n.kind=='constant' and n.output=='foldable' for n in result.graph.ops))
        limited=PassManager((ConstantFolding(max_bytes=0),)).run(w.graph,w.samples)
        self.assertEqual(limited.graph.fingerprint(),w.graph.fingerprint())

    def test_dce_preserves_observable_outputs_and_inputs(self):
        w,result=self.run_pass('dead_branch',DeadCodeElimination())
        self.assertNotIn('unused',[n.output for n in result.graph.ops])
        observable=replace(w.graph,outputs=w.graph.outputs+('unused',))
        kept=PassManager((DeadCodeElimination(),)).run(observable,w.samples)
        self.assertIn('unused',[n.output for n in kept.graph.ops])
        self.assertEqual(kept.graph.inputs,observable.inputs)

    def test_safe_algebraic_rules_and_special_values(self):
        w,result=self.run_pass('redundant_relu',AlgebraicSimplification())
        special=(torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.,-1.,1.,1e-38]),)
        self.assertTrue(validate(w.graph,result.graph,(special,)).passed)
        # x+0 may change signed zero; never simplify it.
        g=native({'version':1,'nodes':[{'name':'x','op':'input','shape':[1],'dtype':'float32'},
            {'name':'zero','op':'constant','dtype':'float32','value':0.},
            {'name':'y','op':'add','inputs':['x','zero']}],'outputs':['y']})
        r=PassManager((AlgebraicSimplification(),)).run(g,((torch.tensor([-0.]),),))
        self.assertEqual(len(r.graph.compute),1)

    def test_layout_inverse_permutations(self):
        w,result=self.run_pass('layout',LayoutSimplification())
        self.assertEqual(result.graph.outputs,('matrix',))
        # Different permutations cannot be cancelled.
        g=replace(w.graph,ops=tuple(replace(n,attrs={'perm':[0,1]}) if n.output=='output' else n for n in w.graph.ops))
        # Metadata now contradicts the permutation, so analysis rejects it.
        with self.assertRaises(Diagnostic): infer_shapes(g)

    def test_fusion_broadcast_and_shared_values(self):
        self.run_pass('affine',AffineReLUFusion())
        w=build('residual',8)
        result=PassManager((AffineReLUFusion(),)).run(w.graph,w.samples)
        self.assertEqual(result.graph.fingerprint(),w.graph.fingerprint())
        class Broadcast(nn.Module):
            def forward(self,x,s,b):return torch.relu(x*s+b)
        for shape in ((0,3),(1,3),(4,3)):
            args=(torch.randn(shape),torch.randn(3),torch.tensor(.1))
            g=pytorch(Broadcast(),args)
            c=compile_graph(g,(args,))
            compare_outputs((Broadcast()(*args),),c.plan(*args))
            self.assertEqual(len(c.graph.compute),1)

    def test_multiple_chains_and_observable_intermediates(self):
        class Chains(nn.Module):
            def forward(self,x,s,b):
                first=torch.relu(x*s+b)
                second=torch.relu(first*s+b)
                return first, second
        args=(torch.randn(4,3),torch.randn(3),torch.tensor(.25))
        graph=pytorch(Chains(),args)
        result=compile_graph(graph,(args,))
        self.assertEqual([n.kind for n in result.graph.compute],['fused_affine_relu']*2)
        compare_outputs(Chains()(*args),result.plan(*args))
        class Shared(nn.Module):
            def forward(self,x,s,b):
                product=x*s
                return torch.relu(product+b), product
        graph=pytorch(Shared(),args)
        result=compile_graph(graph,(args,))
        self.assertNotIn('fused_affine_relu',[n.kind for n in result.graph.compute])

    def test_fusion_extreme_values(self):
        w=build('affine',8)
        sample=(torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.,1e-38,1e38,-1e38]),)
        result=compile_graph(w.graph,(sample,))
        self.assertTrue(result.validation.passed)

    def test_no_samples_rejects_rewrite(self):
        w=build('affine',8)
        r=PassManager().run(w.graph)
        self.assertEqual(r.graph.fingerprint(),w.graph.fingerprint())
        self.assertFalse(validate(w.graph,w.graph,()).passed)
        self.assertTrue(any('no samples' in rec.reason for rec in r.records))

    def test_failed_transformation_rolls_back(self):
        class Wrong:
            name='incorrect'
            def apply(self,g):
                nodes=tuple(replace(n,kind='identity',inputs=(g.inputs[0],)) if n.output==g.outputs[0] else n for n in g.ops)
                return PassResult(replace(g,ops=nodes),1,'unsafe rewrite')
        w=build('affine',8)
        r=PassManager((Wrong(),)).run(w.graph,w.samples)
        self.assertEqual(r.graph.fingerprint(),w.graph.fingerprint())
        self.assertEqual(r.records[0].status,'failed')

    def test_iteration_bound_and_cycle(self):
        class Rename:
            name='rename'
            def apply(self,g):
                tail=g.compute[-1]
                new=tail.output+'_again'
                return PassResult(replace(g,ops=g.ops[:-1]+(replace(tail,output=new),),outputs=(new,)),1,'rename')
        w=build('affine',8)
        r=PassManager((Rename(),),max_iterations=2).run(w.graph,w.samples)
        self.assertFalse(r.converged)
        class Cycle:
            name='cycle'
            def apply(self,g):
                old=g.outputs[0]; new='output' if old=='other' else 'other'
                return PassResult(replace(g,ops=g.ops[:-1]+(replace(g.ops[-1],output=new),),outputs=(new,)),1,'cycle')
        r=PassManager((Cycle(),)).run(w.graph,w.samples)
        self.assertTrue(any('Repeated graph' in rec.reason for rec in r.records))


class FrontendTests(unittest.TestCase):
    def test_full_linear_mlp_and_parameter_snapshot(self):
        torch.manual_seed(9)
        model=nn.Sequential(nn.Linear(3,5),nn.ReLU(),nn.Linear(5,2)).eval()
        args=(torch.randn(4,3),)
        g=pytorch(model,args)
        expected=model(*args).detach()
        compare_outputs((expected,),CPUBackend().compile(g)(*args))
        with torch.no_grad(): model[0].weight.zero_()
        compare_outputs((expected,),CPUBackend().compile(g)(*args))
        self.assertEqual(len(g.compute),7) # two transposes, two matmuls, two adds, relu

    def test_reduction_literals_multiple_outputs(self):
        class Reduction(nn.Module):
            def forward(self,x):return (x*2.0).sum(dim=-1,keepdim=True), x-1.0
        args=(torch.randn(2,3),)
        g=pytorch(Reduction(),args)
        compare_outputs(Reduction()(*args),CPUBackend().compile(g)(*args))

    def test_unsupported_fx_and_native(self):
        with self.assertRaisesRegex(Diagnostic,'unsupported'):
            pytorch(nn.Sigmoid(),(torch.randn(4),))
        with self.assertRaises(Diagnostic): native({'version':2})
        with self.assertRaises(Diagnostic):native({'version':1,'nodes':[{'name':'x','op':'input','dtype':'float16'}],'outputs':['x']})

    def test_onnx_missing_dependency_diagnostic(self):
        import builtins
        real_import=builtins.__import__
        def blocked(name,*args,**kwargs):
            if name=='onnx':raise ImportError('blocked optional dependency')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=blocked):
            with self.assertRaisesRegex(Diagnostic,'optional dependencies'): onnx('absent.onnx')


class PipelineTests(unittest.TestCase):
    def test_all_workloads_both_frontends(self):
        for source in ('native','torch'):
            for name in NAMES:
                with self.subTest(source=source,name=name):
                    w=build(name,8,source=source)
                    for strategy in ('baseline','conventional','manual'):
                        c=compile_graph(w.graph,w.samples,strategy=strategy,
                            passes=('dead_code_elimination',) if strategy=='manual' else None)
                        self.assertTrue(c.validation.passed)
                        self.assertTrue(validate(w.graph,c.graph,(w.evaluation_inputs,)).passed)

    def test_intelligent_measurements_and_report(self):
        w=build('affine',8)
        config=MeasureConfig(1,3,2)
        report=run_suite(((w,'evaluation'),),config)
        row=report['workloads'][0]
        c=row['strategies']['intelligent']
        self.assertTrue(c['heldout_validation']['passed'])
        self.assertEqual(len(c['execution']['raw_ms']),3)
        self.assertGreaterEqual(c['planning_ms'],c['optimization_ms'])
        self.assertIn(c['candidate'],('baseline','conventional','cleanup_only','fusion_only'))
        with tempfile.TemporaryDirectory() as directory:
            raw,human=save(report,Path(directory)/'report.json')
            self.assertEqual(json.loads(raw.read_text())['schema_version'],1)
            self.assertIn('Limitations',human.read_text())

    def test_fallback_and_invalid_configuration(self):
        w=build('affine',8)
        c=compile_graph(w.graph,strategy='intelligent')
        self.assertEqual(c.candidate,'baseline')
        self.assertFalse(c.validation.passed)
        for config in ((-1,3,2),(0,2,1),(0,3,0)):
            with self.assertRaises(ValueError):MeasureConfig(*config)
        with self.assertRaises(ValueError):compile_graph(w.graph,w.samples,strategy='manual',passes=('unknown',))
        with self.assertRaises(ValueError):compile_graph(w.graph,w.samples,strategy='baseline',passes=('affine_relu_fusion',))

    def test_measured_selection_margin(self):
        # Unit-test observations only: these never enter benchmark artifacts or training.
        w=build('affine',8)
        for baseline,winner,mad,expected in ((1.,.8,.001,'conventional'),
                                            (1.,.98,.001,'baseline'),
                                            (1.,.8,.2,'baseline'),
                                            (1.,1.2,.001,'baseline')):
            observations={'timings':{'baseline':{'median_ms':baseline,'mad_ms':mad},
                                     'conventional':{'median_ms':winner,'mad_ms':mad}}}
            with patch('mlcompiler.planner.measure_many',return_value=observations):
                result=compile_graph(w.graph,w.samples,strategy='intelligent')
            self.assertEqual(result.candidate,expected)

    def test_injected_backend_used_for_checks_and_execution(self):
        class ObservedBackend:
            name='observed_cpu'
            def __init__(self):self.calls=0
            def compile(self,graph):
                self.calls+=1
                return CPUBackend().compile(graph)
        w=build('affine',8)
        backend=ObservedBackend()
        result=compile_graph(w.graph,w.samples,backend=backend)
        self.assertGreater(backend.calls,3)
        self.assertTrue(result.validation.passed)

    def test_nan_signed_zero_comparator(self):
        compare_outputs((torch.tensor([float('nan')]),),(torch.tensor([float('nan')]),))
        with self.assertRaises(AssertionError): compare_outputs((torch.tensor([-0.]),),(torch.tensor([0.]),))


@unittest.skipUnless(importlib.util.find_spec('onnx'), 'optional ONNX dependency not installed')
class ONNXTests(unittest.TestCase):
    def model(self,nodes,initializers=(),outputs=None):
        import onnx as ox
        h=ox.helper
        graph=h.make_graph(nodes,'test',[h.make_tensor_value_info('x',ox.TensorProto.FLOAT,[2,3])],
            outputs or [h.make_tensor_value_info('y',ox.TensorProto.FLOAT,[2,3])],list(initializers))
        return h.make_model(graph,opset_imports=[h.make_opsetid('',18)])

    def load(self,model):
        import onnx as ox
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'test.onnx'
        ox.save(model,path)
        return onnx(path),path

    def test_affine_against_onnx_reference(self):
        import onnx as ox
        from onnx.reference import ReferenceEvaluator
        h=ox.helper
        model=self.model([h.make_node('Mul',['x','s'],['p']),h.make_node('Add',['p','b'],['a']),h.make_node('Relu',['a'],['y'])],
            [h.make_tensor('s',ox.TensorProto.FLOAT,[],[1.5]),h.make_tensor('b',ox.TensorProto.FLOAT,[],[.5])])
        g,path=self.load(model)
        args=(torch.randn(2,3),)
        expected=tuple(torch.from_numpy(v) for v in ReferenceEvaluator(str(path)).run(None,{'x':args[0].numpy()}))
        c=compile_graph(g,(args,))
        compare_outputs(expected,c.plan(*args))
        self.assertTrue(c.validation.passed)

    def test_transpose_reshape_matmul_and_rejection(self):
        import onnx as ox
        h=ox.helper
        model=self.model([h.make_node('Transpose',['x'],['t']),h.make_node('Transpose',['t'],['y'])])
        g,path=self.load(model)
        args=(torch.randn(2,3),)
        compare_outputs(args,CPUBackend().compile(g)(*args))
        bad=self.model([h.make_node('Sigmoid',['x'],['y'])])
        with self.assertRaisesRegex(Diagnostic,'unsupported operator'):self.load(bad)
        shape=h.make_tensor('shape',ox.TensorProto.INT64,[2],[3,2])
        model=self.model([h.make_node('Reshape',['x','shape'],['y'])],[shape],
            [h.make_tensor_value_info('y',ox.TensorProto.FLOAT,[3,2])])
        g,_=self.load(model)
        compare_outputs((args[0].reshape(3,2),),CPUBackend().compile(g)(*args))

    def test_full_mlp_reference_and_dynamic_shape(self):
        import onnx as ox
        from onnx.reference import ReferenceEvaluator
        h=ox.helper
        initializers=[h.make_tensor('w',ox.TensorProto.FLOAT,[3,4],[.1]*12),
                      h.make_tensor('v',ox.TensorProto.FLOAT,[4,2],[.2]*8),
                      h.make_tensor('b',ox.TensorProto.FLOAT,[4],[.3]*4)]
        model=self.model([h.make_node('MatMul',['x','w'],['dense']),
                          h.make_node('Add',['dense','b'],['affine']),
                          h.make_node('Relu',['affine'],['hidden']),
                          h.make_node('MatMul',['hidden','v'],['y'])],initializers,
                          [h.make_tensor_value_info('y',ox.TensorProto.FLOAT,['batch',2])])
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param='batch'
        graph,path=self.load(model)
        for size in (0,1,5):
            args=(torch.randn(size,3),)
            expected=tuple(torch.from_numpy(v) for v in ReferenceEvaluator(str(path)).run(None,{'x':args[0].numpy()}))
            result=compile_graph(graph,(args,))
            compare_outputs(expected,result.plan(*args))


if __name__=='__main__':unittest.main()

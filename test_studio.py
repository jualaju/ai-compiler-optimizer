"""Studio serialization, real application-service behavior and HTTP contracts."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import torch
from rich.console import Console
from mlcompiler import Graph, Node, TensorSpec, pytorch
from mlcompiler.workloads import build
from studio.serialization import graph_diff, serialize_graph
from studio.service import StudioService
from studio.server import StudioServer
from terminal_controls import choice, navigate


class SerializationTests(unittest.TestCase):
    def test_edges_match_all_ir_operands_and_outputs(self):
        graph=build('residual',8).graph
        result=serialize_graph(graph)
        ids={n['data']['id'] for n in result['nodes']}
        self.assertEqual(len(result['edges']),sum(len(n.inputs) for n in graph.ops)+len(graph.outputs))
        for edge in result['edges']:
            data=edge['data']
            self.assertIn(data['source'],ids);self.assertIn(data['target'],ids)
        by_id={n['data']['value_id']:n['data'] for n in result['nodes'] if n['data']['node_type']!='output'}
        self.assertEqual(by_id['product']['shape'],[8])
        self.assertEqual(by_id['product']['dtype'],'torch.float32')
        product_uses=[e['data']['target'] for e in result['edges'] if e['data']['tensor_id']=='product']
        self.assertEqual(set(product_uses),{'v:shift','v:output'})

    def test_unknown_metadata_duplicate_operands_multiple_output_aliases(self):
        graph=Graph((Node('x','input'),Node('twice','add',('x','x'))),('twice','x','twice'))
        result=serialize_graph(graph)
        self.assertEqual(result['nodes'][0]['data']['shape_label'],'unknown')
        self.assertEqual(result['nodes'][0]['data']['dtype'],'unknown')
        self.assertEqual(len([n for n in result['nodes'] if n['data']['node_type']=='output']),3)
        self.assertEqual(len({e['data']['id'] for e in result['edges']}),5)

    def test_parameter_roles_and_sources(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU()).eval()
        graph=pytorch(model,(torch.ones(2,3),))
        result=serialize_graph(graph)
        params=[n['data'] for n in result['nodes'] if n['data']['node_type'] in ('weight','bias')]
        self.assertEqual({n['node_type'] for n in params},{'weight','bias'})
        for n in params:
            self.assertEqual(n['metadata']['storage'],'parameter')
            self.assertTrue(n['metadata']['parameter_name'].startswith('0.'))
            self.assertEqual(n['source'],'FX call_module:0')
            self.assertIn('preview',n['constant'])

    def test_dead_nodes_diff_and_nonfinite_json(self):
        service=StudioService()
        r=service.create({'workload':'dead_branch'})
        unused=next(n['data'] for n in r['original']['nodes'] if n['data']['id']=='v:unused')
        self.assertFalse(unused['contributes_to_output'])
        r=service.action(r['id'],'optimize',{})
        self.assertIn('unused',r['diff']['removed'])
        graph=Graph((Node('nan','constant',spec=TensorSpec((),torch.float32),value=torch.tensor(float('nan'))),),('nan',))
        encoded=json.dumps(serialize_graph(graph),allow_nan=False)
        self.assertIn('nan',encoded)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.service=StudioService()

    def test_five_required_examples_and_step_snapshots(self):
        for name,pass_name in (('constant','constant_folding'),('dead_branch','dead_code_elimination'),
                               ('affine','affine_relu_fusion'),('mlp',None),('residual',None)):
            with self.subTest(name=name):
                r=self.service.create({'workload':name,'size':8})
                self.assertIsNone(r['optimized']);self.assertIsNone(r['compilation'])
                r=self.service.action(r['id'],'optimize',{})
                self.assertTrue(r['compilation']['sample_validation']['passed'])
                if pass_name:
                    changed=[s for s in r['steps'] if s['record']['name']==pass_name and s['record']['status']=='applied']
                    self.assertTrue(changed)
                    self.assertFalse(changed[0]['diff']['identical'])
                else:
                    self.assertTrue(r['diff']['identical'])
                    self.assertEqual(r['original']['analysis']['compute_ops'],r['optimized']['analysis']['compute_ops'])
                    self.assertTrue(all(p['status']=='skipped' for p in r['compilation']['passes']))
                if name=='residual':self.assertTrue(any('shared' in p['reason'] for p in r['compilation']['passes']))
                r=self.service.action(r['id'],'validate',{'rtol':0,'atol':0})
                self.assertTrue(r['validation']['passed'])

    def test_manual_policy_and_real_benchmark(self):
        r=self.service.create({'workload':'constant','size':8})
        r=self.service.action(r['id'],'optimize',{'strategy':'manual','passes':['constant_folding']})
        self.assertEqual(r['optimized']['analysis']['compute_ops'],3)
        self.assertEqual(r['compilation']['strategy'],'manual')
        r=self.service.action(r['id'],'benchmark',{'batches':3,'iterations':2,'warmup':0})
        self.assertEqual(r['benchmark']['optimized_fingerprint'],r['optimized']['fingerprint'])
        self.assertEqual(r['benchmark']['selected_policy'],'manual')
        self.assertTrue(r['benchmark']['correctness']['passed'])
        for timing in r['benchmark']['timings'].values():
            self.assertEqual(len(timing['raw_ms']),3)
            self.assertTrue(all(t>0 for t in timing['raw_ms']))

    def test_intelligent_trace_indexes_match_candidate_records(self):
        r=self.service.create({'workload':'constant','size':8})
        r=self.service.action(r['id'],'optimize',{'strategy':'intelligent','batches':3,'iterations':1,'warmup':0})
        for step in r['steps']:
            record=r['compilation']['passes'][step['record_index']]
            self.assertEqual(step['candidate'],record['candidate'])
            self.assertEqual(step['record'],record)
        self.assertGreater(len({s['candidate'] for s in r['steps']}),1)

    def test_errors_preserve_last_good_run_and_are_in_history(self):
        r=self.service.create({'workload':'affine','size':8})
        r=self.service.action(r['id'],'optimize',{})
        fingerprint=r['optimized']['fingerprint']
        with self.assertRaises(ValueError):self.service.action(r['id'],'optimize',{'strategy':'manual','passes':['bad']})
        with self.assertRaises(ValueError):self.service.action(r['id'],'benchmark',{'batches':2})
        actual=self.service.snapshot(self.service.get(r['id']))
        self.assertEqual(actual['optimized']['fingerprint'],fingerprint)
        self.assertEqual(self.service.history()['events'][0]['status'],'failed')
        for data in ({'workload':'bad'},{'size':True},{'size':-1},{'frontend':'unsupported'}):
            with self.assertRaises(ValueError):self.service.create(data)

    def test_native_json_and_explicit_unsupported(self):
        graph=json.loads(Path('examples/native_affine.json').read_text())
        inputs=json.loads(Path('examples/native_inputs.json').read_text())
        r=self.service.create({'frontend':'native_json','graph':graph,'inputs':inputs})
        self.assertFalse(r['reference_checked'])
        r=self.service.action(r['id'],'optimize',{})
        self.assertTrue(r['compilation']['sample_validation']['passed'])
        graph['nodes'][-1]['op']='sigmoid'
        with self.assertRaisesRegex(ValueError,'unsupported'):self.service.create({'frontend':'native_json','graph':graph,'inputs':inputs})


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=StudioServer(('127.0.0.1',0))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def request(self,path,body=None,headers=None):
        req=Request(self.url+path,data=None if body is None else json.dumps(body).encode(),
                    headers=headers or {'Content-Type':'application/json'})
        with urlopen(req,timeout=30) as response:return response.status,response.read()

    def test_api_full_flow_and_static_offline_assets(self):
        for path in ('/','/app.js','/style.css','/vendor/cytoscape.min.js','/api/catalog','/api/support'):
            self.assertEqual(self.request(path)[0],200)
        status,payload=self.request('/api/runs',{'workload':'affine','size':8})
        self.assertEqual(status,201);run=json.loads(payload)
        status,payload=self.request(f'/api/runs/{run["id"]}/optimize',{})
        self.assertEqual(status,200);self.assertEqual(json.loads(payload)['optimized']['analysis']['compute_ops'],1)
        self.assertEqual(self.request(f'/api/runs/{run["id"]}/validate',{})[0],200)
        self.assertEqual(self.request('/api/runs')[0],200)

    def test_useful_errors_and_local_origin(self):
        for path,body,headers,expected in (
            ('/api/runs',{'workload':'bad'},None,400),('/api/runs/absent',None,None,404),
            ('/api/runs',{}, {'Content-Type':'application/json','Origin':'https://elsewhere.invalid'},403),
            ('/api/runs',{}, {'Content-Type':'text/plain'},415),
            ('/%2e%2e/compiler.py',None,None,404)):
            with self.subTest(path=path),self.assertRaises(HTTPError) as error:self.request(path,body,headers)
            self.assertEqual(error.exception.code,expected)
            self.assertIn('error',json.loads(error.exception.read()))


class TerminalTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'PTY navigation requires a POSIX terminal')
    def test_raw_keys_and_failed_action_recovery(self):
        import pty
        program = """from rich.console import Console
from terminal_controls import browse
def fail(): raise ValueError('intentional callback failure')
browse(Console(),lambda:'OVERVIEW',{'d':lambda:'DETAIL CONTENT','b':fail},titles={'d':'Details','b':'Benchmark'})
"""
        master,slave=pty.openpty()
        process=subprocess.Popen([sys.executable,'-c',program],stdin=slave,stdout=slave,stderr=slave)
        os.close(slave)
        output=b''
        def read_until(marker):
            nonlocal output
            deadline=time.monotonic()+10
            while marker not in output and time.monotonic()<deadline:
                if select.select([master],[],[],.1)[0]:
                    try:output+=os.read(master,65536)
                    except OSError:break
            self.assertIn(marker,output)
        try:
            read_until(b'OVERVIEW')
            os.write(master,b'dxnb')
            read_until(b'intentional callback failure')
            os.write(master,b'dq')
            process.wait(timeout=10)
            self.assertEqual(process.returncode,0)
            self.assertIn(b'Unknown key',output)
            self.assertIn(b'DETAIL CONTENT',output)
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=5)
            os.close(master)

    def test_every_displayed_navigation_key_and_unknown_feedback(self):
        views={'o','g','d','p','s','b'}
        for key in views:
            self.assertEqual(navigate(key,'overview',0,views,'a','u')[:2],(key,0))
        self.assertEqual(navigate('n','d',0,views,'a','u')[:2],('d',1))
        self.assertEqual(navigate('u','d',1,views,'a','u')[:2],('d',0))
        self.assertEqual(navigate('a','d',3,views,'a','u')[:2],('overview',0))
        self.assertIn('Unknown key',navigate('x','d',0,views,'a','u')[2])
        self.assertEqual(navigate('q','d',0,views,'a','u')[2],'quit')
        for key in ('k','d'):
            self.assertEqual(navigate(key,'overview',0,{'k','d'},'g','p')[0],key)
        self.assertEqual(navigate('p','d',2,{'k','d'},'g','p')[1],1)

    def test_invalid_menu_selection_is_retried(self):
        out=io.StringIO();console=Console(file=out)
        with patch.object(console,'input',side_effect=['invalid','3']):
            self.assertEqual(choice(console,'Model',('1','2','3'),'2'),'3')
        self.assertIn('Invalid selection',out.getvalue())


if __name__=='__main__':unittest.main()

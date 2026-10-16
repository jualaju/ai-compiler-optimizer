"""Optional real-browser integration checks; no browser dependency for core tests.

Run: .venv/bin/python tests/studio_browser.py
Install: .venv/bin/python -m pip install -r requirements-dev.txt
         .venv/bin/python -m playwright install chromium
"""
from pathlib import Path
import json
import sys
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from playwright.sync_api import sync_playwright, expect
from studio.server import StudioServer


class StudioBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.server=StudioServer(('127.0.0.1',0))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True,executable_path=cls.playwright.chromium.executable_path)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.playwright.stop()
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def setUp(self):
        self.context=self.browser.new_context(viewport={'width':1550,'height':1100})
        self.page=self.context.new_page();self.page.set_default_timeout(10000)
        self.errors=[];self.external=[]
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.page.on('console',lambda m:self.errors.append(m.text) if m.type=='error' and 'status of 400' not in m.text else None)
        self.page.on('request',lambda r:self.external.append(r.url) if not r.url.startswith(self.url) and not r.url.startswith('data:') else None)
        self.page.goto(self.url)
        expect(self.page.locator('#workload')).to_have_value('constant')

    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.assertEqual(self.external,[],'Browser must not depend on external services/CDNs')
        self.context.close()

    def capture(self,name='constant'):
        self.page.locator('#workload').select_option(name)
        self.page.get_by_role('button',name='Capture workload',exact=True).click()
        expect(self.page.locator('#optimize')).to_be_enabled(timeout=30000)
        expect(self.page.locator('#run-label')).to_contain_text(name)

    def optimize(self):
        self.page.get_by_role('button',name='Run optimization',exact=True).click()
        expect(self.page.locator('#validate')).to_be_enabled(timeout=30000)
        expect(self.page.locator('#status')).to_have_text('Compile optimization policy completed.')

    def test_full_browser_pipeline_graphs_and_measurements(self):
        p=self.page
        self.capture();self.optimize()
        p.get_by_role('button',name='Validate outputs',exact=True).click()
        expect(p.locator('#status')).to_have_text('Validate numerical outputs completed.')
        p.get_by_role('tab',name='Graph Explorer').click()
        expect(p.locator('#graph-original canvas').first).to_be_visible()
        expect(p.locator('#graph-optimized canvas').first).to_be_visible()
        before=p.evaluate("() => document.getElementById('graph-original')._cyreg.cy.zoom()")
        p.get_by_role('button',name='Zoom +',exact=True).click()
        after=p.evaluate("() => document.getElementById('graph-original')._cyreg.cy.zoom()")
        self.assertGreater(after,before)
        p.get_by_role('button',name='Zoom −',exact=True).click()
        p.get_by_role('button',name='Fit both',exact=True).click()
        p.get_by_role('button',name='Reset layout',exact=True).click()
        # Click an actual canvas-rendered IR node, not a hard-coded HTML substitute.
        position=p.evaluate("() => document.getElementById('graph-original')._cyreg.cy.getElementById('v:output').renderedPosition()")
        p.locator('#graph-original').scroll_into_view_if_needed()
        box=p.locator('#graph-original').bounding_box()
        p.mouse.click(box['x']+position['x'],box['y']+position['y'])
        expect(p.locator('#node-details')).to_contain_text('relu')
        highlighted=p.evaluate("() => document.getElementById('graph-optimized')._cyreg.cy.getElementById('v:output').hasClass('linked')")
        self.assertTrue(highlighted)
        p.locator('#find-node').fill('foldable')
        p.locator('#node-navigator button').first.click()
        expect(p.locator('#node-details')).to_contain_text('folded_constant')
        p.screenshot(path='/tmp/compiler-studio-graphs.png',full_page=True)
        p.get_by_role('tab',name='Optimization Passes').click()
        applied=p.locator('tbody tr').filter(has=p.locator('td.applied')).first
        expect(applied).to_contain_text('constant_folding')
        applied.get_by_role('button',name='Inspect step').click()
        expect(p.locator('#graph-explanation')).to_contain_text('constant_folding')
        expect(p.locator('#left-title')).to_have_text('Before pass')
        p.get_by_role('tab',name='IR Inspector').click()
        expect(p.locator('#content')).to_contain_text('fused_affine_relu')
        p.get_by_role('tab',name='Compare',exact=True).click()
        p.locator('#batches').fill('3');p.locator('#iterations').fill('2');p.locator('#warmup').fill('0')
        p.get_by_role('button',name='Benchmark original vs selected').click()
        expect(p.locator('#status')).to_have_text('CPU benchmark completed.',timeout=30000)
        expect(p.locator('#content')).to_contain_text('Median ms/call')
        expect(p.locator('#content')).to_contain_text('Raw observations')
        with p.expect_download() as download:
            p.get_by_role('button',name='Download this run as JSON').click()
        self.assertTrue(download.value.suggested_filename.endswith('.json'))
        p.get_by_role('tab',name='Support Matrix').click()
        expect(p.locator('#content')).to_contain_text('Unsupported capabilities')
        p.get_by_role('tab',name='Run History').click()
        expect(p.locator('#content')).to_contain_text('benchmark')
        p.get_by_role('button',name='Refresh history',exact=True).click()
        expect(p.locator('#status')).to_have_text('Refresh history completed.')
        p.get_by_role('button',name='Open run',exact=True).first.click()
        expect(p.locator('#page-title')).to_have_text('Overview')
        p.get_by_role('tab',name='Overview',exact=True).focus()
        p.keyboard.press('ArrowDown')
        expect(p.locator('#page-title')).to_have_text('Graph Explorer')

    def test_manual_noop_error_recovery_and_mobile(self):
        p=self.page
        self.capture('mlp');self.optimize()
        p.get_by_role('tab',name='Graph Explorer').click()
        expect(p.locator('#graph-explanation')).to_contain_text('Graphs are identical')
        p.locator('#strategy').select_option('manual')
        for checkbox in p.locator('#pass-controls input').all():checkbox.uncheck()
        self.optimize()
        p.get_by_role('tab',name='Optimization Passes').click()
        expect(p.locator('#content')).to_contain_text('No passes requested')
        # Workload edits cannot silently optimize the previous captured graph.
        p.locator('#workload').select_option('affine')
        expect(p.locator('#optimize')).to_be_disabled()
        expect(p.locator('#status')).to_contain_text('Workload inputs changed')
        self.capture('affine')
        p.locator('#strategy').select_option('conventional');self.optimize()
        p.get_by_role('tab',name='Compare',exact=True).click()
        p.locator('#batches').fill('2')
        p.get_by_role('button',name='Benchmark original vs selected').click()
        expect(p.locator('#status')).to_contain_text('batches must be an integer from 3 to 31')
        expect(p.locator('#optimize')).to_be_enabled()
        p.locator('#batches').fill('3');p.locator('#iterations').fill('1');p.locator('#warmup').fill('0')
        p.get_by_role('button',name='Benchmark original vs selected').click()
        expect(p.locator('#status')).to_have_text('CPU benchmark completed.',timeout=30000)
        p.locator('#frontend').select_option('native_json')
        p.get_by_role('button',name='Load affine JSON example').click()
        p.get_by_role('button',name='Capture workload',exact=True).click()
        expect(p.locator('#optimize')).to_be_enabled()
        self.optimize()
        p.locator('#native-graph').fill('{invalid json')
        p.get_by_role('button',name='Capture workload',exact=True).click()
        expect(p.locator('#status')).to_contain_text('Capture workload failed')
        # Last successful graph remains inspectable even after a failed capture.
        p.get_by_role('tab',name='Graph Explorer').click()
        expect(p.locator('#graph-original canvas').first).to_be_visible()
        self.page.set_viewport_size({'width':390,'height':844})
        p.screenshot(path='/tmp/compiler-studio-mobile.png',full_page=True)
        overflow=p.evaluate('() => document.documentElement.scrollWidth > window.innerWidth')
        self.assertFalse(overflow,'Mobile layout should not require horizontal page scrolling')

    def test_large_actual_graph_dependencies_pan_and_search(self):
        p=self.page
        nodes=[{'name':'x','op':'input','shape':[4],'dtype':'float32'},
               {'name':'one','op':'constant','dtype':'float32','value':1.0}]
        previous='x'
        for index in range(30):
            name=f'add_{index}'
            nodes.append({'name':name,'op':'add','inputs':[previous,'one']})
            previous=name
        graph={'version':1,'name':'large_chain','nodes':nodes,'outputs':[previous,'add_14']}
        p.locator('#frontend').select_option('native_json')
        p.locator('#native-graph').fill(json.dumps(graph));p.locator('#native-inputs').fill('{"x":[1,2,3,4]}')
        p.get_by_role('button',name='Capture workload',exact=True).click()
        expect(p.locator('#optimize')).to_be_enabled(timeout=30000)
        self.optimize()
        p.get_by_role('tab',name='Graph Explorer').click()
        count=p.evaluate("() => document.getElementById('graph-original')._cyreg.cy.nodes().length")
        self.assertEqual(count,34)  # 32 actual IR values + two distinct output endpoints
        p.get_by_role('button',name='Fit both',exact=True).click()
        p.locator('#find-node').fill('add_29')
        p.locator('#node-navigator button').first.click()
        expect(p.locator('#node-details')).to_contain_text('add_28, one')
        start=p.evaluate("() => document.getElementById('graph-original')._cyreg.cy.pan()")
        p.locator('#graph-original').scroll_into_view_if_needed()
        box=p.locator('#graph-original').bounding_box()
        # Drag empty canvas to pan the real graph viewport.
        p.mouse.move(box['x']+20,box['y']+20);p.mouse.down()
        p.mouse.move(box['x']+90,box['y']+70,steps=5);p.mouse.up()
        end=p.evaluate("() => document.getElementById('graph-original')._cyreg.cy.pan()")
        self.assertNotEqual(start,end)


if __name__=='__main__':unittest.main(verbosity=2)

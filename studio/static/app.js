'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? 'unknown').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pretty = value => JSON.stringify(value, null, 2);
  const ms = value => typeof value === 'number' ? `${value.toFixed(3)} ms` : 'not measured';
  const titles = {overview:'Overview',graphs:'Graph Explorer',ir:'IR Inspector',passes:'Optimization Passes',compare:'Compare',support:'Support Matrix',history:'Run History'};
  const state = {run:null,catalog:null,support:null,history:null,tab:'overview',step:null,graphs:[],busy:false,selected:null,dirty:false};
  const card = (title, body) => `<section class="card"><h2>${esc(title)}</h2>${body}</section>`;
  const pre = value => `<pre>${esc(typeof value === 'string' ? value : pretty(value))}</pre>`;
  function status(message, type='') { $('status').textContent=message; $('status').className=`status ${type}`; }
  function availability() {
    $('capture').disabled=state.busy||!state.catalog;
    $('optimize').disabled=state.busy||!state.run||state.dirty;
    $('validate').disabled=state.busy||!state.run?.compilation||state.dirty;
    document.querySelectorAll('.controls input,.controls select,.controls textarea').forEach(e=>e.disabled=state.busy);
    document.querySelectorAll('[data-server-action]').forEach(e=>e.disabled=state.busy||!state.run?.compilation||state.dirty);
  }
  async function api(path, body) {
    const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||`Request failed (${response.status})`);
    return data;
  }
  async function operation(label, fn) {
    if(state.busy) {status('A compiler operation is still running. Wait for it to finish.','loading');return;}
    state.busy=true;availability();status(`${label}…`,'loading');
    try {await fn();status(`${label} completed.`);}
    catch(error) {status(`${label} failed: ${error.message}`,'error');}
    finally {state.busy=false;try{state.history=await api('/api/runs');}catch(error){status(`History refresh failed: ${error.message}`,'error');}availability();}
  }
  function numeric(id) {
    const raw=$(id).value.trim();
    if(raw==='') throw new Error(`${id} is required`);
    const value=Number(raw);if(!Number.isFinite(value))throw new Error(`${id} must be a finite number`);
    return value;
  }
  const tolerances=()=>({rtol:numeric('rtol'),atol:numeric('atol')});
  function markDirty() {
    if(state.run){state.dirty=true;availability();status('Workload inputs changed. Capture workload to compile these inputs; the displayed graph is still '+state.run.workload+'.');}
  }
  function restoreControls(run) {
    const config=run.configuration||{};
    $('frontend').value=config.frontend||'native';$('workload').value=config.workload||'constant';
    $('size').value=config.size??32;$('seed').value=config.seed??7;
    if(config.graph)$('native-graph').value=pretty(config.graph);
    if(config.inputs)$('native-inputs').value=pretty(config.inputs);
    $('onnx-path').value=config.path||'';$('onnx-inputs').value=config.inputs_path||'';
    if(run.compilation){$('strategy').value=run.compilation.strategy;
      if(run.compilation.strategy==='manual'){
        const used=new Set(run.compilation.passes.map(p=>p.name));
        document.querySelectorAll('#pass-controls input').forEach(input=>input.checked=used.has(input.value));
      }
    }
    $('manual-passes').hidden=$('strategy').value!=='manual';frontendFields();state.dirty=false;
  }
  function frontendFields() {
    const value=$('frontend').value;
    $('native-fields').hidden=value!=='native_json';$('onnx-fields').hidden=value!=='onnx';
    $('workload-field').hidden=!['native','torch'].includes(value);
    description();
  }
  function description() {
    const workload=state.catalog?.workloads.find(w=>w.id===$('workload').value);
    $('workload-description').textContent=['native','torch'].includes($('frontend').value)?workload?.description||'':
      $('frontend').value==='native_json'?'Native JSON is parsed by the existing frontend. No Python source is evaluated.':'ONNX paths are read on the machine running this local server.';
  }
  function setTab(tab, focus=false) {
    state.tab=tab;
    document.querySelectorAll('[data-tab]').forEach(b=>{const active=b.dataset.tab===tab;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;if(focus&&active)b.focus();});
    $('page-title').textContent=titles[tab];render();
  }
  function metric(label,value,note) {return `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`;}
  function empty() {return card('Start with a real workload','<div class="empty"><h2>Capture → inspect → optimize → validate → measure</h2><p>Choose a frontend and workload above, then select Capture workload.</p><p>The native constant-folding example exercises several real passes. The MLP deliberately remains a no-rewrite control.</p></div>');}
  function overview() {
    const r=state.run,c=r.compilation;
    const metrics=metric('Compute operations',`${r.original.analysis.compute_ops} → ${r.optimized?.analysis.compute_ops??'—'}`,'IR structure, not speedup')+
      metric('Planning',c?ms(c.planning_ms):'not run','Includes optimization and calibration')+
      metric('Backend preparation',c?ms(c.backend_preparation_ms):'not run','CPU execution-plan compilation')+
      metric('Numerical check',r.validation?(r.validation.passed?'PASS':'FAILED'):c?.sample_validation.passed?'SAMPLES PASS':'not tested',r.validation?'Evaluation input check':'Compile samples only; evaluate explicitly');
    const stages=[['Frontend preparation',ms(r.frontend_preparation_ms)],['Analysis','Topology + static metadata checked'],['Selected policy',c?`${c.strategy} / ${c.candidate}`:'not compiled'],['Final compile-sample validation',c?ms(c.final_validation_ms):'not run'],['Total common-IR compilation',c?ms(c.total_compile_ms):'not run'],['Execution latency',r.benchmark?`${ms(r.benchmark.timings.optimized.median_ms)} median`:'not benchmarked']];
    return `<div class="metrics">${metrics}</div><div class="grid-two">${card('Workload',`<h3>${esc(r.workload)} <span class="badge">${esc(r.frontend)}</span></h3><p>${esc(r.description)}</p><p class="muted">Backend: ${esc(r.backend)}<br>Frontend reference: ${r.reference_checked?'independently checked':'not independently checked'}<br>Inputs: ${esc(r.input_scope)}</p><div class="note">${esc(r.gpu)}<br>CPU fusion groups dispatch; intermediate tensors remain.</div>`)}${card('Compilation stages',`<div class="stages">${stages.map(([a,b])=>`<div class="stage"><b>${esc(a)}</b><span>${esc(b)}</span></div>`).join('')}</div>`)}</div>`+
      card('Compiler decision',c?`<p>${esc(c.decision)}</p><p class="muted">Optimization time ${ms(c.optimization_ms)} is included in planning; do not add it again.</p><button data-open="passes">Inspect pass decisions</button> <button data-open="graphs">Explore graph changes</button>`:'<p>The original graph has been captured and executed. Run optimization to produce an optimized graph and numerical validation record.</p>');
  }
  function graphExplorer() {
    const r=state.run;
    return `<div class="toolbar"><label class="step-picker">Transformation view<select id="graph-step"><option value="">Original ↔ selected optimized graph</option>${r.steps.map((s,i)=>`<option value="${i}" ${state.step===i?'selected':''}>${esc(s.candidate)} · round ${s.record.iteration} · ${esc(s.record.name)} [${esc(s.record.status)}]</option>`).join('')}</select></label><button id="fit-graphs">Fit both</button><button id="zoom-in">Zoom +</button><button id="zoom-out">Zoom −</button><button id="relayout">Reset layout</button><label>Find a value<input id="find-node" placeholder="Node ID or operation"></label></div><div class="legend"><span>Input / operation</span><span class="constant">Constant</span><span class="weight">Weight / bias</span><span class="output">Graph output</span><span class="changed">Green border: changed</span></div><p class="muted" id="graph-explanation"></p><div class="graph-workspace"><div class="graph-pair"><div class="graph-card"><div class="graph-title"><b id="left-title">Original</b><span>Pan · wheel to zoom</span></div><div id="graph-original" class="cy" aria-label="Original computation graph"></div></div><div class="graph-card"><div class="graph-title"><b id="right-title">Optimized</b><span>Click a node for details</span></div><div id="graph-optimized" class="cy" aria-label="Optimized computation graph"></div></div></div><aside class="card node-details"><h2>Node details</h2><div id="node-details"><p class="muted">Select a node in either graph. Matching value IDs are highlighted in both views. Selecting an output highlights its contributing ancestors.</p></div><div class="details-section"><h3>Node navigator</h3><div id="node-navigator" class="node-list"></div></div></aside></div>`;
  }
  function currentPair() {
    const s=state.step===null?null:state.run.steps[state.step];
    return s?[s.before,s.after]:[state.run.original,state.run.optimized];
  }
  const cyStyle=[
    {selector:'node',style:{'shape':'round-rectangle','width':170,'height':84,'label':'data(label)','text-wrap':'wrap','text-max-width':165,'font-family':'monospace','font-size':13,'color':'#dfeaf4','text-valign':'center','background-color':'#223a4c','border-color':'#547d92','border-width':1.5}},
    {selector:'node[node_type="input"]',style:{'shape':'round-tag','background-color':'#214748','border-color':'#7bd9d0'}},
    {selector:'node[node_type="output"]',style:{'shape':'round-tag','background-color':'#3b3055','border-color':'#c0a4f5'}},
    {selector:'node[node_type="constant"]',style:{'background-color':'#2c3446','border-color':'#8b99b2','width':125}},
    {selector:'node[node_type="weight"],node[node_type="bias"]',style:{'background-color':'#493b2a','border-color':'#e1af70','width':125}},
    {selector:'node[change="modified"],node[change="added"]',style:{'border-width':3,'border-color':'#8cddb0'}},
    {selector:'node[change="removed"]',style:{'border-width':2,'border-color':'#ef9292','border-style':'dashed'}},
    {selector:'node[!contributes_to_output]',style:{'border-style':'dashed','opacity':.65}},
    {selector:'edge',style:{'width':1.4,'curve-style':'bezier','line-color':'#5b7188','target-arrow-color':'#5b7188','target-arrow-shape':'triangle','label':'data(label)','font-size':8,'font-family':'monospace','color':'#adbed0','text-background-color':'#101824','text-background-opacity':.9,'text-background-padding':3,'text-rotation':'autorotate','text-margin-y':-6}},
    {selector:'.ancestor',style:{'line-color':'#7bd9d0','target-arrow-color':'#7bd9d0','border-color':'#7bd9d0','border-width':3}},
    {selector:'node.linked',style:{'border-width':4,'border-color':'#ffffff','overlay-color':'#7bd9d0','overlay-opacity':.12}},
    {selector:'node:selected',style:{'border-width':4,'border-color':'#ffffff'}}
  ];
  function layout(cy) {cy.layout({name:'breadthfirst',directed:true,spacingFactor:1.4,padding:35,avoidOverlap:true,nodeDimensionsIncludeLabels:true,roots:cy.nodes().filter(n=>n.indegree()===0)}).run();}
  function showNode(data) {
    state.selected=data.id;
    state.graphs.forEach(cy=>{
      cy.elements().removeClass('linked ancestor');
      const n=cy.getElementById(data.id);
      if(n.length){n.addClass('linked');n.predecessors().addClass('ancestor');}
    });
    const rows=[['ID',data.value_id],['Node type',data.node_type],['Operation',data.operation],['Inputs',data.inputs.join(', ')||'none'],['Outputs',data.outputs.join(', ')||'none'],['Shape',data.shape_label],['Dtype',data.dtype],['Change',data.change],['Observable',data.contributes_to_output?'contributes to graph output':'dead / does not contribute'],['Source',data.source]];
    $('node-details').innerHTML=`<dl>${rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl><div class="details-section"><h3>Operation attributes</h3>${pre(data.attributes)}</div><div class="details-section"><h3>Constant / parameter metadata</h3>${pre(data.metadata)}${data.constant?pre(data.constant):'<p class="muted">No constant tensor stored on this node.</p>'}</div>`;
  }
  function initGraphs() {
    if(typeof cytoscape!=='function') {status('Local Cytoscape asset failed to load. Check studio/static/vendor.','error');return;}
    const [before,after]=currentPair();
    const snapshot=state.step===null?null:state.run.steps[state.step];
    $('left-title').textContent=snapshot?'Before pass':'Original';$('right-title').textContent=snapshot?'After pass':'Selected optimized';
    $('graph-explanation').textContent=snapshot?`${snapshot.record.name}: ${snapshot.record.reason} ${snapshot.selected_candidate?'This candidate was selected.':'Candidate trace; this may not be the final selected policy.'}`:
      !after?'Optimization has not run. The right view is empty.':state.run.diff.identical?'Graphs are identical. No accepted structural changes were made.':'Nodes are matched by preserved IR value IDs. Red dashed borders mark removed values; green marks changed or added values.';
    [before,after].forEach((graph,i)=>{
      const container=$(i===0?'graph-original':'graph-optimized');
      if(!graph){container.innerHTML='<p class="empty">Run optimization to inspect its actual graph.</p>';return;}
      const cy=cytoscape({container,elements:[...graph.nodes,...graph.edges],style:cyStyle,minZoom:.04,maxZoom:3,layout:{name:'preset'}});
      state.graphs.push(cy);layout(cy);
      cy.on('tap','node',event=>showNode(event.target.data()));
      cy.on('tap','edge',event=>{const source=cy.getElementById(event.target.data('source'));showNode(source.data());});
    });
    function navigator(query='') {
      const all=new Map();[before,after].filter(Boolean).forEach(g=>g.nodes.forEach(n=>all.set(n.data.id,n.data)));
      const matches=[...all.values()].filter(n=>`${n.value_id} ${n.operation}`.toLowerCase().includes(query.toLowerCase()));
      $('node-navigator').innerHTML=matches.map(n=>`<button data-node="${esc(n.id)}">${esc(n.value_id)} · ${esc(n.operation)}</button>`).join('')||'<p class="muted">No matching values.</p>';
      $('node-navigator').querySelectorAll('button').forEach(button=>button.onclick=()=>{
        const data=all.get(button.dataset.node);showNode(data);
        state.graphs.forEach(cy=>{const n=cy.getElementById(data.id);if(n.length){cy.center(n);cy.zoom({level:Math.max(.65,cy.zoom()),renderedPosition:{x:cy.width()/2,y:cy.height()/2}});}});
      });
    }
    navigator();$('find-node').oninput=e=>navigator(e.target.value);
    $('graph-step').onchange=e=>{state.step=e.target.value===''?null:Number(e.target.value);render();};
    $('fit-graphs').onclick=()=>state.graphs.forEach(cy=>{cy.resize();cy.fit(undefined,35);});
    $('zoom-in').onclick=()=>state.graphs.forEach(cy=>cy.zoom(cy.zoom()*1.25));
    $('zoom-out').onclick=()=>state.graphs.forEach(cy=>cy.zoom(cy.zoom()/1.25));
    $('relayout').onclick=()=>state.graphs.forEach(layout);
  }
  function irInspector() {
    const r=state.run;
    const view=graph=>graph?`${pre(graph.ir)}<details><summary>Inputs, outputs and complete node metadata</summary>${pre({inputs:graph.inputs,outputs:graph.outputs,nodes:graph.nodes.map(n=>n.data)})}</details>`:'<p class="muted">No optimized graph yet.</p>';
    return `<div class="grid-two">${card('Original common IR',view(r.original))}${card('Optimized common IR',view(r.optimized))}</div>`;
  }
  function passes() {
    const c=state.run.compilation;
    if(!c)return card('Optimization has not run','<p>Choose a policy and run optimization. There are no fabricated pass results to show.</p>');
    return card('Pass decisions',`<p>${esc(c.decision)}</p><p class="muted">Applicable means an exact graph change was proposed. Applied transformations passed sample checks. Skipped/failed steps retain the actual accepted graph.</p><div class="table-wrap"><table><thead><tr><th>Policy / round</th><th>Pass</th><th>Applicable</th><th>Result</th><th>Ops</th><th>Reason / validation</th><th>Transformation</th></tr></thead><tbody>${c.passes.map((p,i)=>`<tr><td>${esc(p.candidate)} / ${p.iteration}</td><td><code>${esc(p.name)}</code></td><td>${p.applicable?'yes':'no'}</td><td class="${esc(p.status)}">${esc(p.status)}</td><td>${p.before_ops} → ${p.after_ops}</td><td>${esc(p.reason)}<br><span class="muted">Sample validated: ${p.validated?'yes':'no'}</span></td><td>${state.run.steps.some(s=>s.record_index===i)?`<button data-step="${i}">Inspect step</button>`:'no snapshot'}</td></tr>`).join('')||'<tr><td colspan="7">No passes requested by this policy.</td></tr>'}</tbody></table></div>`);
  }
  function compare() {
    const r=state.run,b=r.benchmark;
    const summary=r.diff?`<div class="grid-two">${card('Graph changes',pre(r.diff))}${card('Correctness',r.validation?pre(r.validation):'<p>Run Validate outputs for an evaluation-input check.</p>')}</div>`:'';
    const controls=`<div class="control-row"><label class="small-field">Warm-up<input id="warmup" type="number" min="0" max="100" value="3"></label><label class="small-field">Batches<input id="batches" type="number" min="3" max="31" value="7"></label><label class="small-field">Calls / batch<input id="iterations" type="number" min="1" max="1000" value="20"></label><button id="benchmark" data-server-action>Benchmark original vs selected</button></div><p class="muted">Validates both graphs before timing. Measures the selected policy above, including a manual policy if chosen. Planning is excluded from execution latency.</p>`;
    const results=b?`<div class="table-wrap"><table><thead><tr><th>Execution</th><th>Median ms/call</th><th>MAD ms</th><th>Std dev ms</th><th>Min–max ms</th></tr></thead><tbody>${Object.entries(b.timings).map(([name,t])=>`<tr><td>${esc(name)}</td><td>${t.median_ms.toFixed(6)}</td><td>${t.mad_ms.toFixed(6)}</td><td>${t.stdev_ms.toFixed(6)}</td><td>${t.min_ms.toFixed(6)}–${t.max_ms.toFixed(6)}</td></tr>`).join('')}</tbody></table></div><p>${esc(b.method)}</p><p class="muted">Selected: ${esc(b.selected_policy)} / ${esc(b.selected_candidate)}. Inputs: ${esc(b.input_scope)}.</p><div class="note">${b.limitations.map(esc).join('<br>')}</div><details><summary>Raw observations, batch order and machine metadata</summary>${pre(b)}</details><button class="download" id="download-results">Download this run as JSON</button>`:'<p>No execution measurements collected for this run.</p>';
    return summary+card('CPU benchmark',controls+results)+(r.compilation?card('Execution plan',pre(r.compilation.plan)):'');
  }
  function support() {
    const s=state.support;if(!s)return '<p>Support information is loading.</p>';
    return `<div class="grid-two">${card('Input frontends',s.frontends.map(f=>`<h3>${esc(f.id)} <span class="badge">${esc(f.status)}</span></h3><p class="muted">${esc(f.detail)}</p>`).join(''))}${card('Execution backends',s.backends.map(b=>`<h3>${esc(b.id)} <span class="badge">${esc(b.status)}</span></h3><p class="muted">${esc(b.detail)}</p>`).join(''))}</div>`+card('Exact operator support matrix',pre(s.documentation))+card('Unsupported capabilities',`<ul>${s.unsupported.map(v=>`<li>${esc(v)}</li>`).join('')}</ul>`);
  }
  function history() {
    const h=state.history;if(!h)return '<p>History is loading.</p>';
    return card('Current server session',`<p class="muted">Real completed and failed actions. History is held in memory and resets when this server stops.</p><button id="refresh-history">Refresh history</button><div class="table-wrap"><table><thead><tr><th>Run</th><th>Workload / frontend</th><th>Created</th><th>State</th><th></th></tr></thead><tbody>${h.runs.map(r=>`<tr><td><code>${esc(r.id)}</code></td><td>${esc(r.workload)} / ${esc(r.frontend)}</td><td>${esc(r.created)}</td><td>${r.compiled?'compiled':'captured'}${r.benchmarked?' · benchmarked':''}</td><td><button data-load-run="${esc(r.id)}">Open run</button></td></tr>`).join('')||'<tr><td colspan="5">No captured runs yet.</td></tr>'}</tbody></table></div>`)+card('Action log',`<div class="table-wrap"><table><thead><tr><th>Time / run</th><th>Action</th><th>Status</th><th>Result</th></tr></thead><tbody>${h.events.map(e=>`<tr><td>${esc(e.at)}<br><code>${esc(e.run_id||'capture failed')}</code></td><td>${esc(e.action)}</td><td class="${esc(e.status)}">${esc(e.status)}</td><td>${esc(e.message)}<details><summary>Recorded details</summary>${pre(e.details)}</details></td></tr>`).join('')||'<tr><td colspan="4">No actions yet.</td></tr>'}</tbody></table></div>`);
  }
  function render() {
    state.graphs.forEach(cy=>cy.destroy());state.graphs=[];
    $('run-label').textContent=state.run?`${state.run.workload} / ${state.run.id}`:'No workload captured';
    $('content').innerHTML=state.tab==='support'?support():state.tab==='history'?history():!state.run?empty():
      ({overview,graphs:graphExplorer,ir:irInspector,passes,compare}[state.tab])();
    $('content').querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>setTab(b.dataset.open));
    $('content').querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{state.step=state.run.steps.findIndex(s=>s.record_index===Number(b.dataset.step));setTab('graphs');});
    $('content').querySelectorAll('[data-load-run]').forEach(b=>b.onclick=()=>operation('Load session run',async()=>{state.run=await api(`/api/runs/${b.dataset.loadRun}`);restoreControls(state.run);state.step=null;setTab('overview');}));
    if($('refresh-history'))$('refresh-history').onclick=()=>operation('Refresh history',async()=>{state.history=await api('/api/runs');render();});
    if(state.tab==='graphs'&&state.run)initGraphs();
    if($('benchmark'))$('benchmark').onclick=()=>operation('CPU benchmark',async()=>{
      const settings={warmup:numeric('warmup'),batches:numeric('batches'),iterations:numeric('iterations'),...tolerances()};
      state.run=await api(`/api/runs/${state.run.id}/benchmark`,settings);render();
    });
    if($('download-results'))$('download-results').onclick=()=>{
      const blob=new Blob([pretty(state.run)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`compiler-run-${state.run.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      status('Run JSON download requested.');
    };
    availability();
  }
  document.querySelectorAll('[data-tab]').forEach(button=>{
    button.onclick=()=>setTab(button.dataset.tab);
    button.onkeydown=event=>{const tabs=Object.keys(titles),i=tabs.indexOf(state.tab);let index=null;
      if(['ArrowRight','ArrowDown'].includes(event.key))index=(i+1)%tabs.length;
      if(['ArrowLeft','ArrowUp'].includes(event.key))index=(i+tabs.length-1)%tabs.length;
      if(event.key==='Home')index=0;if(event.key==='End')index=tabs.length-1;
      if(index!==null){event.preventDefault();setTab(tabs[index],true);}
    };
  });
  $('frontend').onchange=()=>{frontendFields();markDirty();};$('workload').onchange=()=>{description();markDirty();};
  ['size','seed','native-graph','native-inputs','onnx-path','onnx-inputs'].forEach(id=>$(id).addEventListener('input',markDirty));
  $('strategy').onchange=()=>{$('manual-passes').hidden=$('strategy').value!=='manual';};
  $('capture').onclick=()=>operation('Capture workload',async()=>{
    const body={frontend:$('frontend').value,workload:$('workload').value,size:numeric('size'),seed:numeric('seed')};
    if(body.frontend==='native_json'){body.graph=JSON.parse($('native-graph').value);body.inputs=JSON.parse($('native-inputs').value);}
    if(body.frontend==='onnx'){body.path=$('onnx-path').value;body.inputs_path=$('onnx-inputs').value;}
    const run=await api('/api/runs',body);state.run=run;state.dirty=false;state.step=null;setTab('overview');
  });
  $('optimize').onclick=()=>operation('Compile optimization policy',async()=>{
    const strategy=$('strategy').value;const body={strategy,...tolerances()};
    if(strategy==='manual')body.passes=[...document.querySelectorAll('#pass-controls input:checked')].map(e=>e.value);
    state.run=await api(`/api/runs/${state.run.id}/optimize`,body);state.step=null;render();
  });
  $('validate').onclick=()=>operation('Validate numerical outputs',async()=>{
    state.run=await api(`/api/runs/${state.run.id}/validate`,tolerances());render();
    if(!state.run.validation.passed)throw new Error(state.run.validation.message);
  });
  $('load-native-example').onclick=()=>{
    $('native-graph').value=pretty({version:1,name:'manual_affine',nodes:[{name:'x',op:'input',shape:[4],dtype:'float32'},{name:'scale',op:'constant',dtype:'float32',value:1.5},{name:'bias',op:'constant',dtype:'float32',value:.5},{name:'p',op:'multiply',inputs:['x','scale']},{name:'a',op:'add',inputs:['p','bias']},{name:'y',op:'relu',inputs:['a']}],outputs:['y']});
    $('native-inputs').value=pretty({x:[2,-1,3,4]});markDirty();status('Native JSON example loaded into the editor. Capture it to run the real frontend.');
  };
  window.addEventListener('resize',()=>state.graphs.forEach(cy=>cy.resize()));
  async function init() {
    try{
      [state.catalog,state.support,state.history]=await Promise.all([api('/api/catalog'),api('/api/support'),api('/api/runs')]);
      $('workload').innerHTML=state.catalog.workloads.map(w=>`<option value="${esc(w.id)}">${esc(w.id)} — ${esc(w.title)}</option>`).join('');$('workload').value=state.catalog.default_workload;
      $('pass-controls').innerHTML=state.catalog.passes.map(name=>`<label><input type="checkbox" value="${esc(name)}" checked>${esc(name)}</label>`).join('');
      $('session-label').textContent='COMPILER READY';frontendFields();render();status('Connected. Select a workload and capture its real graph.');
    }catch(error){status(`Connection failed: ${error.message}. Reload after starting the local server.`,'error');$('session-label').textContent='CONNECTION FAILED';}
  }
  init();
})();

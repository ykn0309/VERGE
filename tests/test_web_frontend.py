"""Render actual frontend functions in Node with a minimal DOM boundary."""
from experience_helpers import keep_update

import json, subprocess
from pathlib import Path
import pytest
from test_web import _run
from verge.web import load_run


def test_new_ui_renders_proposals_applications_adoption_and_dual_evidence(tmp_path):
    directory, _ = _run(tmp_path)
    data = load_run(directory)
    for sample in data["samples"]:
        judgments = [sample.get("initial_judgment")]+[
            a.get("judgment") for a in sample.get("application_attempts", [])
        ]
        for judgment in filter(None, judgments):
            for verdict in judgment["criterion_judgments"]:
                verdict["chosen_reason"] = "chosen <img src=x onerror=alert(1)>"
                verdict["rejected_reason"] = "rejected evidence"
    js = Path("verge/web_assets/app.js").read_text()
    # Stop before event wiring/refresh; rendering uses persisted v4 trajectories.
    js = js[: js.index("const experiencePreview =")]
    program = """const vm=require('node:vm');const fs=require('node:fs');
const ctx={document:{getElementById:()=>({})},URLSearchParams,location:{search:''},setTimeout,clearTimeout};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
vm.runInContext('data='+fs.readFileSync(process.argv[2],'utf8'),ctx);
const output=vm.runInContext('[overview(),pipelineDetail(data.samples[0]),candidateDetail(candidateEntries()[0]),experienceDetail(data.experiences[0]),samplesPage(),memoryPage()]',ctx);
console.log(JSON.stringify(output));"""
    source = tmp_path / "render.js"
    source.write_text(js)
    payload = tmp_path / "data.json"
    payload.write_text(json.dumps(data))
    result = subprocess.run(
        ["node", "-e", program, str(source), str(payload)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = json.loads(result.stdout)
    assert "候选、应用与入池" in html[0] and "成功但未采用" in html[0]
    assert "新Generator" in html[1] and "声明采用：是" in html[1]
    assert (
        "&lt;script&gt;not code&lt;/script&gt;" in html[1]
        and "<script>not code</script>" not in html[1]
    )
    assert "Critic 原始输出" in html[2]
    assert "P " in html[3] and "/ N " in html[3]
    assert all("Reflector" not in h and "0.25" not in h and "Admission" not in h and "准入" not in h for h in html)

    assert "分配权重 0.2" in html[1] and "Judge 判断依据" in html[1]
    assert "chosen &lt;img src=x onerror=alert(1)&gt;" in html[1]
    assert "<img src=x onerror=alert(1)>" not in html[1]
    assert "rejected evidence" in html[1]
    assert '<details class="criterion-audit"><summary>Judge 判断依据' in html[1]


def test_actual_text_update_events_render_versions_and_escape_text(tmp_path):
    from test_bounded_experience import existing_runner
    from verge.web import _evolution
    r, eid = existing_runner(tmp_path / 'update')
    r.modules.experience_updater.update = lambda e: keep_update(e, '共同方法 <img src=x onerror=alert(1)>', '必要条件')
    r.run_sample(__import__('test_pipeline').sample('warm1'))
    r.run_sample(__import__('test_pipeline').sample('warm2'))
    record = r.run_sample(__import__('test_pipeline').sample('next'))
    experience = r.memory.get(eid).to_dict()
    experience.update(case_count=len(experience['cases']), text_frozen=False)
    experience['evolution'] = _evolution(experience, {}, {'seed': 0, 'next': 1}, [])
    source = tmp_path / 'render.js'
    source.write_text(Path('verge/web_assets/app.js').read_text().split('const experiencePreview =')[0])
    payload = tmp_path / 'payload.json'
    payload.write_text(json.dumps({'experiences': [experience], 'samples': [record.to_dict()]}))
    program = """const vm=require('node:vm'),fs=require('node:fs');
const ctx={document:{getElementById:()=>({})},URLSearchParams,location:{search:''},setTimeout,clearTimeout};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
vm.runInContext('data='+fs.readFileSync(process.argv[2],'utf8'),ctx);
console.log(vm.runInContext('experienceDetail(data.experiences[0])',ctx));"""
    rendered = subprocess.run(['node','-e',program,str(source),str(payload)], capture_output=True, text=True)
    assert rendered.returncode == 0, rendered.stderr
    assert '案例 4/10' in rendered.stdout and '文本 v1 → v2' in rendered.stdout
    assert '文本重新归纳' in rendered.stdout and '查看原文本' in rendered.stdout
    assert '&lt;img src=x onerror=alert(1)&gt;' in rendered.stdout
    assert '<img src=x onerror=alert(1)>' not in rendered.stdout


def test_experience_sorting_preserves_ids_selection_and_scroll(tmp_path):
    """Exercise actual delegated change/click handlers after reordering and searching."""
    js = Path('verge/web_assets/app.js').read_text().removesuffix('refresh();\n')
    source = tmp_path / 'app.js'
    source.write_text(js)
    program = r"""const vm=require('node:vm'), fs=require('node:fs'), assert=require('node:assert/strict');
const listeners={}, list={scrollTop:0};
const element={addEventListener(){},hidden:true,open:false};
const app={querySelector:()=>list, set innerHTML(value){this.html=value;list.scrollTop=0}};
const document={getElementById:id=>id==='app'?app:element,
  addEventListener(type,fn){(listeners[type]??=[]).push(fn)},
  querySelectorAll:()=>[],querySelector:()=>({focus(){}})};
const ctx={document,URLSearchParams,location:{search:''},setTimeout,clearTimeout,
  window:{scrollTo(){throw Error('click unexpectedly scrolled page')}}};
vm.createContext(ctx); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
vm.runInContext(`data={warnings:[],experiences:[.4,.9,.9,.6].map((confidence,i)=>({
  experience_id:'e'+i,content:'method'+i,trigger:'*',confidence,update_history:[],case_count:1,text_version:1})),
  samples:[{sample_id:'s1',initial_rubric:{criteria:[{source_experiences:['e0','e1']},{source_experiences:['e0']}]}},
    {sample_id:'s2',initial_rubric:{criteria:[{source_experiences:['e0']}]},selected_experiences:[{experience_id:'e2'}]},
    {sample_id:'s3',application_attempts:[{rubric:{criteria:[{source_experiences:['e2']}]}}]}]};
page='memory';experienceDetail=e=>'DETAIL:'+e.experience_id;`,ctx);
function order(){return [...app.html.matchAll(/data-action="experience" data-index="(\d+)"/g)].map(m=>Number(m[1]));}
function sort(value){list.scrollTop=240;listeners.change.forEach(fn=>fn({target:{value,matches:()=>true}}));assert.equal(list.scrollTop,240);}
sort('confidence_desc');assert.deepEqual(order(),[1,2,3,0]);
assert.ok(app.html.includes('data-index="0"') && app.html.includes('DETAIL:e0'));
const button={dataset:{action:'experience',index:'2'}};
list.scrollTop=320;listeners.click.forEach(fn=>fn({target:{closest:()=>button}}));
assert.equal(list.scrollTop,320);assert.ok(app.html.includes('DETAIL:e2'));
sort('confidence_asc');assert.deepEqual(order(),[0,3,1,2]);assert.ok(app.html.includes('DETAIL:e2'));
sort('usage_desc');assert.deepEqual(order(),[0,1,2,3]);assert.ok(app.html.includes('引用 2 次'));
sort('usage_asc');assert.deepEqual(order(),[2,3,1,0]);assert.ok(app.html.includes('DETAIL:e2'));
vm.runInContext("expQuery='method1';render()",ctx);assert.deepEqual(order(),[1]);assert.ok(app.html.includes('DETAIL:e1'));
vm.runInContext("expQuery='';memoryTab='candidates';render()",ctx);assert.ok(!app.html.includes('data-experience-sort'));
console.log('sorting, selection, search and scroll passed');"""
    result = subprocess.run(['node', '-e', program, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

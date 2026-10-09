"""Render actual frontend functions in Node with a minimal DOM boundary."""

import json, subprocess
from pathlib import Path
import pytest
from test_web import _run
from verge.web import load_run


def test_new_ui_renders_proposals_applications_adoption_and_dual_evidence(tmp_path):
    directory, _ = _run(tmp_path)
    data = load_run(directory)
    js = Path("verge/web_assets/app.js").read_text()
    # Stop before event wiring/refresh; rendering uses persisted v2 artifacts.
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
    assert "候选、准入与入池" in html[0] and "成功但未采用" in html[0]
    assert "新Generator" in html[1] and "声明采用：是" in html[1]
    assert (
        "&lt;script&gt;not code&lt;/script&gt;" in html[1]
        and "<script>not code</script>" not in html[1]
    )
    assert "Critic 原始输出" in html[2]
    assert "P " in html[3] and "/ N " in html[3]
    assert all("Reflector" not in h and "0.25" not in h for h in html)

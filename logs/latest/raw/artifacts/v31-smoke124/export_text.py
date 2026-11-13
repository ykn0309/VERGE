"""Archive prior evidence without deletion; publish latest evidence as plain text."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path('runs/v31-smoke124');ART=Path('artifacts/v31-smoke124')
LATEST=Path('logs/latest');ARCHIVE=Path('logs/archive/2026-11-13-before124')
LIMIT=512*1024

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def publish():
    assert json.loads((ROOT/'manifest.json').read_text())['status']=='completed'
    if ARCHIVE.exists():
        # Resume only our interrupted text export. Original archived evidence
        # is already hash-verified and must never be moved or overwritten again.
        index=json.loads((LATEST/'archive-index.json').read_text())
        entries=index['moves']
        assert not (ARCHIVE/'partial-text-export').exists()
        shutil.move(str(LATEST),str(ARCHIVE/'partial-text-export'))
    else:
        sources=[LATEST,Path('runs/history-learning-general500-subset120'),Path('artifacts/general500-subset120'),Path('artifacts/general500-cluster-audit121'),Path('artifacts/general500-case-audit122')]
        entries=[]
        for source in sources:
            if not source.exists():continue
            target=ARCHIVE/source
            files=[(str(p.relative_to(source)),sha(p),p.stat().st_size) for p in sorted(source.rglob('*')) if p.is_file()]
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
            assert all(sha(target/rel)==digest for rel,digest,size in files)
            entries.append({'original':str(source),'archive':str(target),'files':len(files),'bytes':sum(n for _,_,n in files),'all_hashes_verified':True})
    LATEST.mkdir(parents=True,exist_ok=False)
    dump(LATEST/'archive-index.json',{'previous_git_commit':'7b9f119','moves':entries,'older_archive':'logs/archive/2026-11-12'})
    manifest=[]
    for base in [ROOT,ART]:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix==".pyc":continue
            data=path.read_bytes();data.decode('utf8')
            destination=LATEST/'raw'/path
            parts=[]
            if len(data)<=LIMIT:
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);parts=[destination]
            else:
                folder=Path(str(destination)+'.parts');folder.mkdir(parents=True)
                chunks=[]
                if path.suffix=='.jsonl':
                    chunk=b''
                    for line in data.splitlines(keepends=True):
                        if chunk and len(chunk)+len(line)>LIMIT:chunks.append(chunk);chunk=b''
                        chunk+=line
                    if chunk:chunks.append(chunk)
                else:
                    rest=data
                    while rest:
                        cut=min(LIMIT,len(rest))
                        while cut and cut<len(rest) and rest[cut]&0xc0==0x80:cut-=1
                        chunks.append(rest[:cut]);rest=rest[cut:]
                for i,chunk in enumerate(chunks,1):
                    part=folder/f'{i:04d}.txt';part.write_bytes(chunk);parts.append(part)
            assert b''.join(p.read_bytes() for p in parts)==data
            manifest.append({'source':str(path),'bytes':len(data),'sha256':sha(path),'parts':[{'path':str(p.relative_to(LATEST)),'sha256':sha(p)} for p in parts]})
    dump(LATEST/'manifest.json',{'files':manifest,'all_reconstructed_hashes_verified':True})
    checkpoint=json.loads((ROOT/'checkpoint.json').read_text())
    pool=checkpoint['experiences'];dump(LATEST/'experience_pool.json',pool)
    lines=['# 入池经验索引','', '来源验证仅允许入池，不代表经验可靠。','', '| 编号 | 经验ID与正文 | 来源样本 |','|---|---|---|']
    samplemap={r['sample']['task']:r['sample']['sample_id'] for r in checkpoint['trajectories']}
    taskmap={__import__('verge.models',fromlist=['task_identity']).task_identity(r['sample']['task']):r['sample']['sample_id'] for r in checkpoint['trajectories']}
    for i,e in enumerate(pool,1):
        name=f'E{i:02d}.json';dump(LATEST/'experiences'/name,e)
        lines.append(f'| E{i:02d} | [{e["experience_id"]}]({name}) | {taskmap[e["source_task_id"]]} |')
    (LATEST/'experiences'/'README.md').write_text('\n'.join(lines)+'\n')
    training={r['sample']['sample_id']:r for r in checkpoint['trajectories']}
    ordering=[json.loads(x)['sample_id'] for x in (ART/'train.jsonl').read_text().splitlines()]
    tests=[json.loads(x) for x in (ROOT/'test.jsonl').read_text().splitlines()]
    records=[training[sid] for sid in ordering]+tests
    lines=['# 逐题轨迹索引','', '训练的各经验路平等试用，不选择事后赢家；测试仅有一条冻结路径。','', '| 编号 | 阶段 | 完整轨迹 | 初始路径结果 | Critic轮数 |','|---|---|---|---|---|']
    def result(p):return '未知' if p['error'] or not p['judgment'] else '对' if p['judgment']['success'] else '错'
    for i,r in enumerate(records,1):
        phase='train' if i<=len(ordering) else 'test';name=f'{phase}-{i if phase=="train" else i-len(ordering):03d}.json'
        dump(LATEST/'samples'/name,r)
        paths='；'.join(('无经验' if not p['provided_experience_ids'] else p['provided_experience_ids'][0])+':'+result(p) for p in r['paths'])
        lines.append(f'| {i} | {phase} | [{r["sample"]["sample_id"]}]({name}) | {paths or "任务未知"} | {len(r["critic_rounds"])} |')
    (LATEST/'samples'/'README.md').write_text('\n'.join(lines)+'\n')
    (LATEST/'README.md').write_text('''# 最新实验：v3.1小样本完整流程124

20题在线训练＋5题冻结推理。代码7b9f119，Qwen3.5-Flash关闭思考，从空经验池开始。这里只验证运行与内容问题，不是泛化效果结论。

- [总体报告124](../../docs/实验分析/124-v3.1小样本真实流程验证.md)
- [逐题轨迹](samples/README.md)：全部25题完整任务、回答、初稿、各路修订、Judge及Critic。
- [经验索引](experiences/README.md)与[全部经验](experience_pool.json)。
- [原始模型请求及回复](raw/runs/v31-smoke124/role_io.jsonl.parts/)：按记录分段，均为UTF-8文本。
- [输入、配置、协议、旁路校验与人工审查](raw/artifacts/v31-smoke124/)。
- [原文件及分段摘要](manifest.json)：顺序拼接parts可恢复原字节，全部校验通过。没有压缩包。

上一轮v3的120／121／122证据已无损移入忽略的logs/archive/2026-11-13-before124，映射见[archive-index.json](archive-index.json)。Git中仍可通过[7b9f119时的旧日志](https://github.com/ykn0309/VERGE/tree/7b9f119/logs/latest)直接读取。数据集输入目录不属于实验日志，不归档。
''')
    print(json.dumps({'archived':entries,'published_raw_files':len(manifest),'experiences':len(pool),'samples':len(records)},ensure_ascii=False))
if __name__=='__main__':publish()

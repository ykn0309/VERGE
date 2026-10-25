// Pure DOM contract test: no browser process, user UI, or network interaction.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element{
 constructor(tag){this.tagName=tag;this.children=[];this.dataset={};this.className='';this.textContent='';this.scrollTop=0;this.value='';this.disabled=false;this.classList={toggle:()=>{}};}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=[...children];}
}
const nodes={'#app':new Element('main'),'#run':new Element('select'),'#refresh':new Element('button')};
const tabs=['overview','samples','experiences'].map(page=>{const x=new Element('button');x.dataset.page=page;return x;});
function walk(root,predicate){if(predicate(root))return root;for(const x of root.children||[]){const hit=walk(x,predicate);if(hit)return hit;}return null;}
const document={createElement:t=>new Element(t),querySelector:s=>nodes[s]||walk(nodes['#app'],x=>s[0]==='.'&&x.className.split(' ').includes(s.slice(1))),querySelectorAll:s=>s==='[data-page]'?tabs:[]};
const data={name:'fixture',manifest:{},metrics:{tasks:50,initial_correct:20,active:50,retired:1},revisions:[],
 experiences:Array.from({length:51},(_,i)=>({experience_id:'e'+i,method:{trigger:'条件'+i,principle:'方法'+i,template:null,parameters:{}},status:i===50?'retired':'active',q_posterior:i===0?null:.7,parent_id:null,q_prior:null,parameter_version:null,applications:[]})),
 samples:Array.from({length:50},(_,i)=>({sample_id:'s'+i,task:'任务'+i,chosen:'优选',rejected:'非优选',state:'initial_success',experience_updates:[],applications:[],proposal_attempts:[]}))};
const context=vm.createContext({document,URLSearchParams,location:{search:''},history:{replaceState:()=>{}},fetch:async path=>({ok:true,json:async()=>path==='/api/runs'?{runs:['fixture'],default:'fixture'}:data}),console});
vm.runInContext(fs.readFileSync('verge/web_assets/app.js','utf8'),context);
setImmediate(()=>{
 vm.runInContext("state.page='experiences';render()",context);
 let sidebar=document.querySelector('.sidebar');sidebar.scrollTop=430;
 const items=sidebar.children.filter(x=>x.tagName==='button');items[15].onclick();
 assert.equal(document.querySelector('.sidebar').scrollTop,430,'experience click must preserve list scroll');
 sidebar=document.querySelector('.sidebar');const filters=sidebar.children[0];const sort=filters.children[1];sort.value='references';sort.onchange();
 assert.equal(vm.runInContext('state.sort',context),'references');
 let status=document.querySelector('.sidebar').children[0].children[0];status.value='retired';status.onchange();
 assert.equal(vm.runInContext('state.experience',context),'e50');
 tabs[1].onclick();sidebar=document.querySelector('.sidebar');sidebar.scrollTop=260;sidebar.children[20].onclick();
 assert.equal(document.querySelector('.sidebar').scrollTop,260,'sample click must preserve list scroll');
 assert.equal(vm.runInContext('state.sample',context),'s20');
 tabs[2].onclick();assert.equal(vm.runInContext('state.page',context),'experiences');
 vm.runInContext("state.status='active';state.experience='e1';state.data.experiences[1].q_prior=.6;state.data.experiences[1].applications=[];render()",context);
 assert.match(vm.runInContext('quality(state.data.experiences[1])',context),/基础先验/);
 function textTree(e){return [e.textContent,...(e.children||[]).map(textTree)].join(' ');}
 const visible=textTree(nodes['#app']);
 for(const text of ['证据来源','下游应用','原始逐标准反馈','双方满足','回放使用归纳看过的案例'])assert.ok(visible.includes(text),text);
 tabs[1].onclick();assert.ok(textTree(nodes['#app']).includes('检索、选择与排名'));
 assert.ok(textTree(nodes['#app']).includes('加载原始角色请求／回复'));
 console.log('DOM contract checks passed: list scroll, category removal, status/sort controls and navigation.');
});

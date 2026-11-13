#!/usr/bin/env python3
"""Build offline research views from the existing records; requires installed pandoc."""
import hashlib
import html
import json
import itertools
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = '''
:root{color-scheme:light;--ink:#172e38;--muted:#52656c;--accent:#0072B2;--paper:#f6f5ef;--line:#cbd4d4}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,"Microsoft YaHei","Noto Sans CJK SC",sans-serif}
:lang(zh){line-height:1.85;line-break:strict;text-autospace:normal;letter-spacing:0}body{font-size:1.0625rem;overflow-wrap:anywhere}p{max-width:65ch}a{color:#005e91;text-underline-offset:.18em;overflow-wrap:anywhere}a:hover{color:#943f00}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #0072B2;outline-offset:4px}
header,main,footer{max-width:1240px;margin:auto;padding:2rem clamp(1rem,4vw,3rem)}header{border-bottom:1px solid var(--line)}nav{display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.9rem}h1{font-size:clamp(2rem,4.5vw,3.8rem);line-height:1.25;font-weight:650;max-width:20em}h2{margin-top:2.8rem;font-size:1.65rem;line-height:1.4}h3{line-height:1.4}small,.muted,figcaption{color:var(--muted)}.kicker{font-size:.8rem;letter-spacing:.12em;color:var(--accent)}.lead{font-size:1.2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:1.2rem}.card,details,figure{background:white;border:1px solid var(--line);padding:1.2rem;margin:1rem 0;border-radius:4px}.big{font-size:2.8rem;font-variant-numeric:tabular-nums;line-height:1.2}.tag{font-size:.8rem;color:var(--muted)}.best{border-top:4px solid var(--accent)}summary{cursor:pointer;font-weight:600}details>div{margin-top:1rem}code,pre{font-size:.85em;overflow-wrap:anywhere}pre{overflow:auto;background:#edf1f2;padding:1rem}.table-wrapper{overflow:auto;max-inline-size:100%;max-block-size:70vh;container-type:inline-size}table{border-collapse:separate;border-spacing:0;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums;text-align:start}th,td{padding:.65rem .8rem;border-bottom:1px solid #dee4e4;vertical-align:top;min-width:6rem}th{background:#edf2f3}thead{position:sticky;inset-block-start:0;z-index:2}tbody th{position:sticky;inset-inline-start:0;z-index:1;background:#fff}caption{text-align:start;font-weight:600;padding:.5rem}figure{margin-inline:0}figcaption{font-family:Georgia,"Noto Serif CJK SC",serif;font-size:.82rem;text-align:start;margin-top:1rem}em{font-style:normal;text-emphasis:filled dot}math{font-size:1.1em;overflow-x:auto;text-autospace:no-autospace}input,select,button{font:inherit;padding:.5rem .7rem;border:1px solid #81979f;background:white;color:var(--ink);border-radius:3px}.controls{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.barrow{display:grid;grid-template-columns:minmax(12rem,1fr) 2fr 5rem;gap:1rem;align-items:center;margin:.6rem 0;font-size:.9rem}.track{height:.8rem;background:#e9edf0}.bar{height:100%;background:#0072B2}.diagram{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.node{padding:1rem;background:#edf3f5;border-left:3px solid #0072B2}.merge{text-align:center;padding:1rem;background:#172e38;color:white}.history-item{scroll-margin-top:1rem}.history-item h2{margin-top:0}.skip{position:absolute;left:-9999px}.skip:focus{left:1rem;top:1rem;background:white;padding:1rem}footer{font-size:.8rem;border-top:1px solid var(--line)}[hidden]{display:none!important}
@media(max-width:650px){.diagram{grid-template-columns:1fr}.barrow{grid-template-columns:1fr 5rem}.barrow .track{grid-column:1/-1;grid-row:auto}h1{font-size:2rem}header,main,footer{padding:1rem}}
@media print{body{background:white;color:black;font-size:10pt}nav,.controls,.skip,button{display:none}header,main,footer{max-width:none;padding:.5cm}figure,math,.card{break-inside:avoid}h2,h3{break-after:avoid}a[href^="http"]::after{content:" (" attr(href) ")";font-size:8pt;overflow-wrap:anywhere}.table-wrapper{overflow:visible;max-block-size:none}thead,tbody th{position:static}table{table-layout:fixed;font-size:8pt}tr{break-inside:avoid}th,td{min-width:0;padding:.2rem;overflow-wrap:anywhere}pre{white-space:pre-wrap}details{break-inside:auto}details::details-content{display:block}a{color:black}.merge{background:white;color:black;border:1px solid black}}
'''
CSS += r"""

/* A compact research workbench; full records stay in the same offline page. */
header,main,footer{max-width:1600px}header{padding-block:1.2rem}header h1{font-size:2rem;margin:.5rem 0}header .lead{font-size:1rem;margin:.5rem 0}header p:last-child{font-size:.88rem;margin:0}h2{margin-top:1.6rem}.workbench{margin:1.4rem 0 2rem}.workbench-heading{display:flex;justify-content:space-between;align-items:end;gap:1rem;flex-wrap:wrap}.workbench-heading h2{margin:0}.workbench-heading p{margin:.3rem 0;font-size:.88rem}.table-toolbar{display:flex;gap:.7rem;flex-wrap:wrap;padding:1rem;background:#e9eff1;border:1px solid var(--line);border-bottom:0}.table-toolbar label{display:grid;gap:.25rem;font-size:.76rem;color:var(--muted)}.table-toolbar input,.table-toolbar select{max-width:100%;font-size:.88rem;min-height:2.5rem}.table-toolbar .search{flex:1;min-width:min(100%,220px)}.table-toolbar .range{width:7.5rem}.table-status{display:flex;gap:1rem;justify-content:space-between;align-items:center;padding:.55rem .8rem;font-size:.8rem;border:1px solid var(--line);border-top:0;background:white}.result-grid{max-block-size:650px;border:1px solid var(--line);background:white}.result-grid table{min-width:1200px;font-size:.82rem}.result-grid th,.result-grid td{padding:.65rem .7rem;min-width:0;white-space:nowrap}.result-grid tbody tr:nth-child(even){background:#f6f8f9}.result-grid tbody tr.selected{background:#e6f2fa}.result-grid thead th{padding:0}.result-grid thead button{width:100%;min-height:44px;border:0;border-radius:0;background:transparent;text-align:start;font-size:.8rem;font-weight:650;white-space:nowrap}.result-grid th[aria-sort=ascending] .sort-icon::after{content:' ↑'}.result-grid th[aria-sort=descending] .sort-icon::after{content:' ↓'}.result-grid th:not([aria-sort]) .sort-icon::after{content:' ↕';opacity:.45}.result-grid td:nth-child(2){position:sticky;inset-inline-start:0;z-index:1;background:inherit;min-width:280px;max-width:370px;white-space:normal}.result-grid tbody{background:white}.model-button{border:0;padding:0;background:transparent;color:#005e91;text-align:start;font-size:.86rem;font-weight:650;line-height:1.5}.model-button:hover{text-decoration:underline}.result-grid .numeric{font-variant-numeric:tabular-nums;text-align:end}.source-label{display:block;max-width:17rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.badge{display:inline-block;padding:.15rem .45rem;border-radius:3px;font-size:.72rem;background:#edf1f3;color:#41575f}.badge.candidate{background:#d9edf9;color:#005680}.metric-cell{display:flex;align-items:center;justify-content:end;gap:.5rem}.mini-bar{width:44px;height:4px;background:#dce5eb}.mini-bar i{display:block;height:100%;background:#0072B2}.inspector{padding:1.2rem 1.4rem;background:white;border:1px solid var(--line);border-top:3px solid #0072B2;margin-top:1rem}.inspector-heading{display:flex;gap:1rem;justify-content:space-between;align-items:start}.inspector h3{margin:0}.inspector p{font-size:.9rem;max-width:none}.inspector-metrics{display:flex;gap:1.5rem;flex-wrap:wrap;margin:.6rem 0}.inspector-metrics strong{font-size:1.25rem;font-variant-numeric:tabular-nums}.flow-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1.5rem;counter-reset:step}.flow-step{position:relative;padding:.8rem;border:1px solid #cbdee9;background:#f1f7fa;font-size:.85rem;counter-increment:step}.flow-step::before{content:'0' counter(step);display:block;color:#0072B2;font-weight:700;font-size:.7rem}.flow-step:not(:last-child)::after{content:'→';position:absolute;right:-1.25rem;top:40%;font-size:1.2rem;color:#0072B2}.diagram-viewport{overflow:auto;max-width:100%}.candidate-svg{display:block;min-width:760px;width:100%;height:auto}.candidate-svg text{font-family:inherit;fill:#172e38}.candidate-svg .title{font-size:18px;font-weight:700}.candidate-svg .label{font-size:15px}.candidate-svg .note{font-size:13px;fill:#52656c}.candidate-svg .arrow{fill:none;stroke:#687f8a;stroke-width:2;marker-end:url(#flow-arrow)}.diagram-controls{display:flex;flex-wrap:wrap;gap:.5rem;margin:.8rem 0}.diagram-controls button{font-size:.85rem}.diagram-controls [aria-pressed=true]{background:#172e38;color:white}.diagram-focus-tabm .kernel-branch,.diagram-focus-huber .neural-branch{opacity:.25}.diagram-focus-fusion .neural-branch,.diagram-focus-fusion .kernel-branch{opacity:.35}.diagram-explanation{background:#edf3f5;padding:.9rem 1.2rem;margin-top:.8rem;border-left:3px solid #0072B2;font-size:.92rem}.legacy-overview{margin-top:1rem}.archive{padding:1rem}.archive>summary{font-size:1.1rem}.empty-state{text-align:center;padding:2rem;border:1px solid var(--line);background:white}.pagination{display:flex;gap:.5rem;align-items:center}.table-status button{padding:.2rem .6rem;font-size:.8rem}.workbench button:disabled{opacity:.4;cursor:not-allowed}
@media(max-width:650px){.flow-strip{grid-template-columns:1fr 1fr}.flow-step::after{display:none}.table-status{align-items:start;flex-direction:column}.inspector{padding:1rem}.inspector-heading{display:block}.result-grid td:nth-child(2){position:static;min-width:220px}.table-toolbar label{max-width:100%}.diagram-controls button{flex:1}.table-toolbar .search{flex-basis:100%}}
@media print{.table-toolbar,.pagination,.diagram-controls{display:none}.result-grid{max-block-size:none}.result-grid table{min-width:0;font-size:7pt}.result-grid td:nth-child(2){min-width:0;max-width:none;position:static}.result-grid th,.result-grid td{white-space:normal}.model-button{display:inline;color:black}.candidate-svg{min-width:0}.flow-strip{break-inside:avoid}.inspector{break-inside:avoid}}
"""

JS = '''
function filterGroup(input, selector) {
 const query=input.value.toLocaleLowerCase();let n=0;
 document.querySelectorAll(selector).forEach(el=>{el.hidden=!el.textContent.toLocaleLowerCase().includes(query);if(!el.hidden)n++});
 document.getElementById(input.dataset.count).textContent=`显示 ${n} 项`;
}
document.querySelectorAll('input[data-filter]').forEach(input=>{filterGroup(input,input.dataset.filter);input.addEventListener('input',()=>filterGroup(input,input.dataset.filter))});
window.addEventListener('beforeprint',()=>{document.querySelectorAll('details').forEach(d=>{d.dataset.wasOpen=d.open;d.open=true})});
window.addEventListener('afterprint',()=>{document.querySelectorAll('details').forEach(d=>{d.open=d.dataset.wasOpen==='true'})});
'''

JS += r"""
(() => {
 const payload = document.getElementById('model-data');
 if (!payload) return;
 const {records, architectures} = JSON.parse(payload.textContent);
 const table = document.getElementById('model-table'), tbody = table.tBodies[0];
 const rows = new Map([...tbody.rows].map(row => [Number(row.dataset.record), row]));
 const byId = new Map(records.map(record => [record.id, record]));
 const collator = new Intl.Collator('zh-CN', {numeric:true, sensitivity:'base'});
 const controls = Object.fromEntries(['model-search','role-filter','stage-filter','family-filter','pcc-min','pcc-max','page-size'].map(id=>[id,document.getElementById(id)]));
 let sortKey='priority', direction=1, pageIndex=0, selected=0, filtered=records;
 const explanations = {
  all:'两条分支分别完成内层选模与完整开发集重拟合，再在预测空间固定等权平均。这里没有联合训练的新融合头，也不根据外层结果改权重。',
  tabm:'神经分支：每个环境单独使用全部 1,279 个标记。多个 MLP 成员共享大部分权重，每成员输出 99 分位点并接受 pinball 监督。分位点和成员平均、反标准化后，再平均种子 42 与 2026 的预测。',
  huber:'核分支：K 描述材料的标记相似性，B 描述拟合材料的环境相关性。ICM 用 B ⊗ K 共享四环境信息；Huber 损失限制大残差的梯度影响，岭惩罚约束复杂度。它是凸优化预测器，不是完整贝叶斯 Huber GP 采样。',
  fusion:'融合：两个分支的点预测都先还原到原表型尺度，再固定各占 50%。本次折均宏 PCC 为 0.526506，TabM 为 0.518829、Huber 为 0.521745；这是完整系统的描述性结果，不单独证明生物学机制。'
 };
 document.querySelectorAll('[data-explain]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-explain]').forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  document.getElementById('candidate-diagram').className='diagram-focus-'+button.dataset.explain;
  document.getElementById('diagram-explanation').textContent=explanations[button.dataset.explain];
 }));
 function inspect(id) {
  selected=id;
  const record=byId.get(id), panel=document.getElementById('model-inspector');
  panel.hidden=!record;
  rows.forEach((row,key)=>row.classList.toggle('selected',key===id));
  if (!record) return;
  const [steps, description, href]=architectures[record.family];
  document.getElementById('inspector-title').textContent=record.name;
  document.getElementById('inspector-family').textContent=record.family+' · '+record.stage+' · '+record.role;
  const doc=document.getElementById('inspector-doc');doc.href=record.role==='主候选'?'#best':href;doc.textContent=record.role==='主候选'?'查看重点图解 ↓':'查看模型方案 ↗';
  const metrics=document.getElementById('inspector-metrics');metrics.replaceChildren();
  ['pcc','rmse','mae'].forEach(key=>{const div=document.createElement('div'), strong=document.createElement('strong');strong.textContent=record[key];div.append(key.toUpperCase()+' ',strong);metrics.append(div)});
  const flow=document.getElementById('inspector-flow');flow.replaceChildren();
  const shownSteps=record.role==='主候选'?['全标记输入','四独立分位数 TabM ＋ Huber ICM','各自反标准化 · 神经双种子平均','固定各占 50% → 四环境预测']:steps;
  shownSteps.forEach(text=>{const step=document.createElement('div');step.className='flow-step';step.textContent=text;flow.append(step)});
  document.getElementById('inspector-description').textContent=description;
  const source=document.getElementById('inspector-source'), link=document.createElement('a');link.href=record.link;link.textContent=record.source;source.replaceChildren('本行原记录：',link);
  if (record.env.some(value=>value!=='—')) source.append(' · Env1–4 PCC：'+record.env.join(' / '));
 }
 function apply() {
  const query=controls['model-search'].value.trim().toLocaleLowerCase();
  const min=controls['pcc-min'].value===''?null:controls['pcc-min'].valueAsNumber;
  const max=controls['pcc-max'].value===''?null:controls['pcc-max'].valueAsNumber;
  filtered=records.filter(record=>{
   if (query && ![record.name,record.source,record.family,record.stage].join(' ').toLocaleLowerCase().includes(query)) return false;
   for (const [control,key] of [['role-filter','role'],['stage-filter','stage'],['family-filter','family']]) if(controls[control].value && record[key]!==controls[control].value) return false;
   if (min!==null || max!==null) {
    const score=record.values.pcc;
    if(score===null || (min!==null && score<min) || (max!==null && score>max))return false;
   }
   return true;
  });
  filtered.sort((a,b)=>{
   const numeric=['pcc','rmse','mae','priority'].includes(sortKey);
   const av=sortKey==='priority'?a.id:numeric?a.values[sortKey]:a[sortKey];
   const bv=sortKey==='priority'?b.id:numeric?b.values[sortKey]:b[sortKey];
   if(av===null)return bv===null?a.id-b.id:1;
   if(bv===null)return -1;
   return direction*(numeric?av-bv:collator.compare(av,bv)) || a.id-b.id;
  });
  const size=Number(controls['page-size'].value)||filtered.length||1, pages=Math.max(1,Math.ceil(filtered.length/size));
  pageIndex=Math.min(pageIndex,pages-1);
  const visible=filtered.slice(pageIndex*size,(pageIndex+1)*size), ids=new Set(visible.map(record=>record.id));
  rows.forEach((row,id)=>row.hidden=!ids.has(id));
  filtered.forEach(record=>{const row=rows.get(record.id);if(tbody.moveBefore)tbody.moveBefore(row,null);else tbody.append(row)});
  document.getElementById('model-empty').hidden=filtered.length!==0;
  document.getElementById('model-count').textContent=`筛选 ${filtered.length} / ${records.length} 条 · 当前显示 ${visible.length} 条`;
  document.getElementById('page-number').textContent=`${pageIndex+1} / ${pages} 页`;
  document.getElementById('previous-page').disabled=pageIndex===0;
  document.getElementById('next-page').disabled=pageIndex===pages-1;
  document.querySelectorAll('#model-table thead th').forEach(th=>{th.removeAttribute('aria-sort');if(th.dataset.key===sortKey)th.setAttribute('aria-sort',direction===1?'ascending':'descending')});
  if (!ids.has(selected)) inspect(visible[0]?.id);
 }
 function sort(button) {
  const key=button.dataset.sort;
  direction=sortKey===key?-direction:1;sortKey=key;pageIndex=0;apply();
 }
 if ('commandForElement' in HTMLButtonElement.prototype) table.addEventListener('command',event=>{if(event.command==='--sort')sort(event.source)});
 else table.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>sort(button)));
 table.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>inspect(Number(button.dataset.model))));
 Object.values(controls).forEach(control=>control.addEventListener(control.tagName==='SELECT'?'change':'input',()=>{pageIndex=0;apply()}));
 document.getElementById('previous-page').addEventListener('click',()=>{pageIndex--;apply()});
 document.getElementById('next-page').addEventListener('click',()=>{pageIndex++;apply()});
 document.getElementById('reset-models').addEventListener('click',()=>{Object.entries(controls).forEach(([id,control])=>control.value=id==='page-size'?'25':'');sortKey='priority';direction=1;pageIndex=0;apply();inspect(0)});
 document.getElementById('export-models').addEventListener('click',()=>{
  const csvCell=value=>{let text=String(value??'');if(/^[=+@\t\r]/.test(text)||(/^\-/.test(text)&&!/^\-\d+(\.\d+)?$/.test(text)))text="'"+text;return '"'+text.replaceAll('"','""')+'"'};
  const header=['模型','类别','架构家族','评价口径','PCC','RMSE','MAE','Env1 PCC','Env2 PCC','Env3 PCC','Env4 PCC','状态','来源','来源链接'];
  const values=filtered.map(r=>[r.name,r.role,r.family,r.stage,r.pcc,r.rmse,r.mae,...r.env,r.status,r.source,r.link]);
  const csv='\ufeff'+[header,...values].map(row=>row.map(csvCell).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const anchor=document.createElement('a');anchor.href=url;anchor.download='wheat599-model-results.csv';document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 document.addEventListener('click',event=>{
  const link=event.target.closest('a[href^="#"]');if(!link)return;
  const target=document.getElementById(decodeURIComponent(link.hash.slice(1)));if(!target)return;
  for(let node=target;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;
 });
 inspect(0);apply();
})();
"""


def esc(value):
    return html.escape(str(value), quote=True)


SECTION_IDS = itertools.count()


def markdown(text):
    rendered = subprocess.run(['pandoc', '-f', 'markdown', '-t', 'html5', '--mathml'],
                              input=text, text=True, capture_output=True, check=True).stdout
    prefix = f'section-{next(SECTION_IDS)}-'
    rendered = re.sub(r' id="([^"]+)"', lambda m: ' id="' + prefix + m[1] + '"', rendered)
    rendered = re.sub(r'href="#([^"]+)"', lambda m: 'href="#' + prefix + m[1] + '"', rendered)
    return re.sub(r'<table([^>]*)>(.*?)</table>',
                  r'<div class="table-wrapper" tabindex="0" role="region" aria-label="记录数据表"><table\1>\2</table></div>',
                  rendered, flags=re.S)


def page(title, body, destination):
    destination.write_text(f'''<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><meta name="description" content="Wheat599 结果、架构和证据的离线研究视图">
<meta name="citation_title" content="{esc(title)}"><style>{CSS}</style></head><body>
<a class="skip" href="#main">跳至内容</a>{body}<footer>Wheat599 项目内研究记录 · 数据与权重不随本页分发。由 scripts/build_report.py 从现有证据生成；修订来源后重建，勿手改生成页。</footer><script>{JS}</script></body></html>''')


ARCHITECTURES = {
    '固定融合': (['全标记输入', '神经预测分支 ＋ 核预测分支', '各自还原表型尺度', '按本行约定进行融合'], '融合家族示意：当前主候选为分位数 TabM 与 Huber ICM 各占 50%；历史 MSE 分支、融合选模与其他配方分别见本行原记录。', 'QUANTILE_TABM_FINAL.md'),
    'TabM': (['拟合集标准化 · 全标记', '共享主权重的 MLP 成员', '单值 MSE 或 99 分位点输出', '成员平均 · 种子预测平均'], 'TabM 家族示意：当前分位数模型为四环境独立；共享、检索、交互与蒸馏变体的实际改动见该行原记录。', 'QUANTILE_TABM.md'),
    'ICM / 核方法': (['拟合集材料相似性核', '核组合 / 多环境任务结构', '正则化求解或稳健拟合', '查询材料的表型预测'], '核方法家族示意：Huber ICM 使用加性＋RBF 材料核及拟合表型相关矩阵，带 Huber 损失和岭惩罚。其他 ICM/LMC 变体的核结构和选模规则分别记录。', 'RESEARCH_PLAN.md'),
    'Mul-PheG2P': (['源与目标环境编码器', '预训练 · 目标适配', '标量或隐藏表示注意力融合', '预测 · 可选折内校准'], '完整适配允许目标编码器参与。标量、隐藏表示及有无校准分列；公开默认与内层选配置分列，不以外层结果择优拼接。', 'MULPHE_UNIFIED.md'),
    'WheatGP': (['原标记 / 拟合集 PCA', '五段标记 · CNN', 'LSTM 序列建模', '回归输出'], '公开 CNN–LSTM 网络的本项目适配；按实际卷积输出长度调整输入维度。原重建流程出现的常数预测和汇总失败保留在表中。', 'WHEATGP.md'),
    'MoEGP': (['全标记输入', '门控网络选择专家', 'MLP 专家预测', '加权专家输出'], '独立初始化专家与官方共享专家引用是两套不同运行；按每行的系统名称与来源识别。', 'WHEATGP_MOEGP_FOLLOWUP.md'),
    'BiMSGP': (['单标记线性嵌入', '正向 / 反向 Mamba', '对齐后拼接并展平', 'MLP 解码器'], '历史已运行的 BiMSGP 适配，不是文件名中 bimsgp 协议的 FT-Transformer。当前不重启该模型。', '../configs/bimsgp_nested.yaml'),
    'DNAwhisper': (['卷积标记嵌入', 'GFI 编码与交叉注意力', '专家与池化模块', '四表型输出'], '拟合集内监督预训练与微调，预训练包含表型；不能称为无标签预训练。', '../configs/dnawhisper_nested.yaml'),
    'RAGP': (['拟合集 PCA95 / 编码', '检索五个拟合材料邻居', '特征与参考标签融合', '残差网络回归'], '按材料 ID 排除自检索，参考库逐轮更新。历史适配误差尺度未校准，需结合来源解释 RMSE/MAE。', '../src/wheat599/ragp.py'),
    'TabICLv2': (['拟合材料作为上下文', '预训练表格模型', '原权重 / 合成任务续训', '查询材料回归'], '当前仅开发比较：官方权重、通用合成任务续训和遗传合成任务续训分开记录。', 'TABICL_GENOMIC.md'),
    'FT-Transformer': (['逐标记 token', '自注意力 / 诱导注意力', '池化或因子化读出', '表型回归'], 'FT 家族示意。A 架构、Direct FT-T、诱导注意力和固定 RR 残差修正分别保留实际配方，A 不指谱系矩阵。', '../configs/bimsgp.yaml'),
    'RR-BLUP': (['拟合集中心化标记', '线性标记效应', 'L2 正则 / 加性核岭', '表型点预测'], '基础线性参照；不同历史运行的训练材料预算、正则选择和划分不能自动合并。', '../src/wheat599/kernel_baselines.py'),
    'RFM': (['标准化全标记', 'Laplace 核回归', 'AGOP 更新特征度量', '重新拟合核预测器'], '四环境独立，完整 1,279×1,279 特征矩阵；配置与迭代次数在开发集内选择。', 'RFM_DEVELOPMENT.md'),
    '稀疏 / 条件网络': (['全标记与环境任务', '条件单元 / 双线性残差', '逐成员 99 分位点', '成员与种子平均'], '两类具体网络须区分：层级条件网络读取先前单元输出；稀疏残差网络使用双线性残差块。完整定义见各行结果与架构库。', 'SPARSE_RESIDUAL_QUANTILE.md'),
    '贝叶斯树': (['全标记线性背景', '环境专属稀疏森林', '联合采样残差协方差', '保留样本预测平均'], '固定预算 Monte Carlo 预测；部分链混合不足，结果不代表已收敛后验或整个模型家族失败。', 'SPARSE_POLYGENIC_BART.md'),
    '其他研究变体': (['该运行的拟合材料', '按具体配置构建模型', '开发集选模 / 固定规则', '对应评价层的结果'], '这一行无法仅凭系统简称确定完整网络；请打开本行原记录，架构库保留其详细方案，不把家族示意冒充实际配置。', '#architectures'),
}


def model_family(name, source):
    def identify(text):
        text = text.lower()
        for token, family in [('mulphe', 'Mul-PheG2P'), ('mul-phe', 'Mul-PheG2P'), ('wheatgp', 'WheatGP'),
                              ('moegp', 'MoEGP'), ('bimsgp', 'BiMSGP'), ('dnawhisper', 'DNAwhisper'),
                              ('ragp', 'RAGP'), ('tabicl', 'TabICLv2'), ('rfm', 'RFM')]:
            if token in text:
                return family
        if ('融合' in text and ('tabm' in text or 'huber' in text or '分位数' in text)) or '预固定0.5' in text:
            return '固定融合'
        if '稀疏' in text and ('树' in text or '森林' in text):
            return '贝叶斯树'
        if '稀疏残差' in text or '层级条件' in text:
            return '稀疏 / 条件网络'
        if 'tabm' in text or '分位数' in text or '蒸馏' in text or '单成员检索网络' in text:
            return 'TabM'
        if any(t in text for t in ['ft-t', 'ft-transformer', 'a 架构', 'a 骨干', '诱导注意力', 'a（']):
            return 'FT-Transformer'
        if 'rr-blup' in text or 'gblup' in text:
            return 'RR-BLUP'
        if any(t in text for t in ['icm', '核', 'kernel', 'lmc', 'lcm', 'huber', 'rbf']):
            return 'ICM / 核方法'
        return None
    family = identify(name)
    if family:
        return family
    if 'TabICL' in source and any(token in name for token in ['官方', '续训', '合成任务']):
        return 'TabICLv2'
    return '其他研究变体'


def comparison_records(systems):
    records = []
    current_source = '已核对统一协议 · 分位数最终评价'
    for index, (name, metrics, environments) in enumerate(systems):
        source = '完整 Mul-PheG2P 同协议适配（2026-11-13）' if 'Mul-Phe' in name else current_source
        records.append(dict(name=name, stage='统一协议外层', pcc=f'{metrics["pcc"]:.6f}',
                            rmse=f'{metrics["rmse"]:.6f}', mae=f'{metrics["mae"]:.6f}',
                            env=[f'{environments[e]["pcc"]:.6f}' for e in ['1','2','4','5']],
                            source=source, link='#run-0' if 'Mul-Phe' in name else 'QUANTILE_TABM_FINAL.md',
                            role='主候选' if index == 0 else '强参照' if index < 3 else '公开模型',
                            priority=(0 if index == 0 else index if index < 3 else 3 if '三内折选配置' in name and '标量融合' in name else 4 if '公开默认配置' in name and '标量融合' in name else 90 + index), primary=True))
    comparison = (ROOT/'docs/MODEL_COMPARISON.md').read_text().split('## 全模型总对比表', 1)[1].split('\n## ', 1)[0]
    source_ids = {block.split('\n',1)[0]: f'#run-{i}' for i, block in enumerate(
        re.split(r'^## ', (ROOT/'docs/RESULTS.md').read_text(), flags=re.M)[1:])}
    public = ['Mul-PheG2P', 'WheatGP', 'MoEGP', 'BiMSGP', 'DNAwhisper', 'RAGP', 'TabICLv2']
    representatives = {'Mul-PheG2P'}
    for line in comparison.splitlines():
        if not line.startswith('| ') or line.startswith('| 评价层'):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) != 6:
            raise ValueError(f'Unexpected model comparison row: {line}')
        stage, name, pcc, rmse, mae, raw_source = cells
        match = re.search(r'\[([^]]+)\]\(([^)]+)\)', raw_source)
        if not match:
            raise ValueError(f'Missing result source: {line}')
        source, link = match.groups()
        name = name.replace('`', '')
        family = model_family(name, source)
        role = '公开模型' if family in public else '基础参照' if family == 'RR-BLUP' else '研究变体 / 历史'
        priority = 100 + public.index(family) if family in public else 500
        # Give each published comparator one visible representative before its historical variants.
        if family in public and family not in representatives and not ('异常' in stage or '重建' in stage):
            priority = 15 + public.index(family)
            representatives.add(family)
        records.append(dict(name=name, stage=stage, pcc=pcc, rmse=rmse, mae=mae, env=['—']*4,
                            source=source, link=source_ids.get(source, link), role=role,
                            priority=priority, primary=False))
    records.sort(key=lambda row: row['priority'])
    for i, record in enumerate(records):
        record['id'] = i
        record['family'] = model_family(record['name'], record['source'])
        record['status'] = '指标无定义' if '无定义' in record['pcc'] else '有记录'
        record['values'] = {}
        for key in ['pcc','rmse','mae']:
            value = record[key].replace('−','-')
            record['values'][key] = float(value) if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', value) else None
    return records


def result_workbench(systems):
    records = comparison_records(systems)
    headers = [('priority','默认序'),('name','模型 / 完整系统'),('role','类别'),('family','架构家族'),
               ('stage','评价口径'),('pcc','PCC ↑'),('rmse','RMSE ↓'),('mae','MAE ↓'),
               ('status','状态'),('source','记录来源')]
    headings = ''.join(f'<th scope="col" data-key="{key}"' + (' aria-sort="ascending"' if key == 'priority' else '') +
                       f'><button type="button" commandfor="model-table" command="--sort" data-sort="{key}">{label}<span class="sort-icon" aria-hidden="true"></span></button></th>' for key,label in headers)
    rows = []
    for record in records:
        number = record['values']['pcc']
        bar = f'<span class="mini-bar" aria-hidden="true"><i style="width:{max(0,number)*100}%"></i></span>' if number is not None else ''
        row = f'<tr data-record="{record["id"]}"><td>{record["id"]+1:03}</td><td><button type="button" class="model-button" data-model="{record["id"]}">{esc(record["name"])}</button></td>'
        row += f'<td><span class="badge {"candidate" if record["role"]=="主候选" else ""}">{esc(record["role"])}</span></td>'
        row += f'<td>{esc(record["family"])}</td><td>{esc(record["stage"])}</td><td class="numeric"><span class="metric-cell">{bar}{esc(record["pcc"])}</span></td>'
        row += ''.join(f'<td class="numeric">{esc(record[key])}</td>' for key in ['rmse','mae'])
        row += f'<td>{esc(record["status"])}</td><td><a class="source-label" href="{esc(record["link"])}" title="{esc(record["source"])}">{esc(record["source"])}</a></td></tr>'
        rows.append(row)
    options = lambda key: '<option value="">全部</option>' + ''.join(f'<option>{esc(value)}</option>' for value in dict.fromkeys(r[key] for r in records))
    data = json.dumps({'records':records, 'architectures':ARCHITECTURES}, ensure_ascii=False).replace('<','\\u003c')
    return f'''<section class="workbench" id="comparison" aria-labelledby="comparison-title">
<div class="workbench-heading"><div><h2 id="comparison-title">模型结果总表</h2><p>默认顺序：主候选 → 强参照与公开模型 → 其余研究。点击模型查看结构，点击表头排序。</p></div><a href="#best">跳到最优候选图解 ↓</a></div>
<div class="table-toolbar">
<label class="search">搜索模型 / 配方 / 来源<input id="model-search" type="search" placeholder="例如：WheatGP、分位数、校准"></label>
<label>类别<select id="role-filter">{options('role')}</select></label>
<label>评价口径<select id="stage-filter">{options('stage')}</select></label>
<label>架构家族<select id="family-filter">{options('family')}</select></label>
<label class="range">PCC 下限<input id="pcc-min" type="number" min="-1" max="1" step="0.01" placeholder="不限"></label>
<label class="range">PCC 上限<input id="pcc-max" type="number" min="-1" max="1" step="0.01" placeholder="不限"></label>
<label>每页<select id="page-size"><option>25</option><option>50</option><option>100</option><option value="0">全部</option></select></label>
</div>
<div class="table-wrapper result-grid" tabindex="0" role="region" aria-labelledby="model-table-caption"><table id="model-table"><caption id="model-table-caption">{len(records)} 条系统记录 · 每行保留评价口径与来源；同一模型的不同运行分别展示</caption><thead><tr>{headings}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div id="model-empty" class="empty-state" hidden>没有匹配记录，请调整筛选条件。</div>
<div class="table-status"><span id="model-count" role="status">{len(records)} 条记录</span><div class="pagination"><button id="reset-models" type="button">恢复默认</button><button id="export-models" type="button">导出筛选结果 CSV</button><button id="previous-page" type="button">上一页</button><span id="page-number"></span><button id="next-page" type="button">下一页</button></div></div>
<p class="muted" style="font-size:.8rem">开发、外层、历史重建与种子指标均值保留原口径。公开模型是本项目的适配结果；请按评价口径筛选后比较。无定义与缺失指标保留原文字，不当作 0。筛选和排序不改变模型选择。</p>
<aside id="model-inspector" class="inspector" aria-labelledby="inspector-title"><div class="inspector-heading"><div><span id="inspector-family" class="tag"></span><h3 id="inspector-title">固定等权分位数–Huber 融合</h3></div><a id="inspector-doc" href="#best">查看完整架构</a></div><div id="inspector-metrics" class="inspector-metrics"></div><div id="inspector-flow" class="flow-strip"></div><p id="inspector-description"></p><p id="inspector-source" class="muted"></p></aside>
<script id="model-data" type="application/json">{data}</script></section>'''


def candidate_diagram():
    return '''<figure id="candidate-diagram">
<div class="diagram-controls" role="group" aria-label="选择图解步骤"><button type="button" data-explain="all" aria-pressed="true">完整预测路径</button><button type="button" data-explain="tabm" aria-pressed="false">① 分位数 TabM</button><button type="button" data-explain="huber" aria-pressed="false">② Huber ICM</button><button type="button" data-explain="fusion" aria-pressed="false">③ 固定等权融合</button></div>
<div class="diagram-viewport" tabindex="0" role="region" aria-label="最优候选的双分支结构图，可横向滚动">
<svg class="candidate-svg" viewBox="0 0 1040 625" role="img" aria-labelledby="candidate-svg-title candidate-svg-description" xmlns="http://www.w3.org/2000/svg">
<title id="candidate-svg-title">四独立分位数 TabM 与多环境 Huber ICM 的固定等权融合</title><desc id="candidate-svg-description">一份新材料的1279个标记进入两条独立训练的分支。神经分支在每个环境中以多个MLP成员输出99分位点，平均并还原尺度后平均两种子。核分支组合材料核和拟合表型环境相关矩阵，进行带岭惩罚的Huber拟合。最终两分支在原表型尺度固定各占一半，输出四个环境预测。</desc>
<defs><marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#687f8a"/></marker></defs>
<rect x="310" y="8" width="420" height="64" rx="8" fill="#172e38"/><text x="520" y="35" text-anchor="middle" class="title" style="fill:white">新材料 · 1,279 个二值标记</text><text x="520" y="57" text-anchor="middle" class="note" style="fill:#d5e1e7">两条分支均沿用当前外折拟合集的预处理</text>
<path class="arrow" d="M520 72 V88 H260 V109"/><path class="arrow" d="M520 88 H780 V109"/>
<g class="neural-branch"><rect x="20" y="112" width="490" height="334" rx="10" fill="#edf6fc" stroke="#80b5d4"/><text x="42" y="142" class="title" style="fill:#005e91">① 分位数 TabM · 四环境独立</text>
<g fill="white" stroke="#80b5d4"><rect x="42" y="161" width="100" height="40" rx="4"/><rect x="155" y="161" width="100" height="40" rx="4"/><rect x="268" y="161" width="100" height="40" rx="4"/><rect x="381" y="161" width="106" height="40" rx="4"/></g>
<text x="92" y="187" text-anchor="middle" class="label">Env1 网络</text><text x="205" y="187" text-anchor="middle" class="label">Env2 网络</text><text x="318" y="187" text-anchor="middle" class="label">Env3 网络</text><text x="434" y="187" text-anchor="middle" class="label">Env4 网络</text>
<path class="arrow" d="M265 202 V224"/><rect x="42" y="229" width="445" height="83" rx="6" fill="white" stroke="#80b5d4"/><text x="265" y="256" text-anchor="middle" class="label">每个网络：共享主权重 × 16 / 32 个 MLP 成员</text><text x="265" y="281" text-anchor="middle" class="label">每成员输出 99 个分位点 · pinball 损失</text><text x="265" y="302" text-anchor="middle" class="note">分位点为 0.001–0.999；成员数不等于训练种子数</text>
<path class="arrow" d="M265 312 V334"/><rect x="42" y="339" width="445" height="80" rx="6" fill="#d8ebf7"/><text x="265" y="365" text-anchor="middle" class="label">分位点与成员平均 → 反标准化</text><text x="265" y="390" text-anchor="middle" class="label">种子 42、2026 的预测再平均</text><text x="265" y="410" text-anchor="middle" class="note">输出：每个环境一个点预测</text></g>
<g class="kernel-branch"><rect x="530" y="112" width="490" height="334" rx="10" fill="#fff4ed" stroke="#d9a07e"/><text x="552" y="142" class="title" style="fill:#96400d">② Huber ICM · 四环境联合</text>
<rect x="552" y="161" width="213" height="70" rx="5" fill="white" stroke="#d9a07e"/><rect x="783" y="161" width="213" height="70" rx="5" fill="white" stroke="#d9a07e"/><text x="659" y="188" text-anchor="middle" class="label">材料核 K</text><text x="659" y="211" text-anchor="middle" class="note">加性核 ＋ RBF 核</text><text x="890" y="188" text-anchor="middle" class="label">环境矩阵 B</text><text x="890" y="211" text-anchor="middle" class="note">拟合表型相关矩阵的收缩</text>
<path class="arrow" d="M659 231 V247 H775 V263"/><path class="arrow" style="marker-end:none" d="M890 231 V247 H775"/><rect x="552" y="268" width="444" height="54" rx="5" fill="white" stroke="#d9a07e"/><text x="775" y="292" text-anchor="middle" class="label">联合信号核 Ω = B ⊗ K</text><text x="775" y="313" text-anchor="middle" class="note">用环境关联共享材料相似性信息</text>
<path class="arrow" d="M775 323 V334"/><rect x="552" y="339" width="444" height="80" rx="6" fill="#f8e4d7"/><text x="775" y="365" text-anchor="middle" class="label">Huber 损失 ＋ 岭惩罚 → 凸优化</text><text x="775" y="390" text-anchor="middle" class="label">四环境联合预测 → 反标准化</text><text x="775" y="410" text-anchor="middle" class="note">大残差转为线性惩罚，不删除材料</text></g>
<path class="arrow" d="M265 446 V470 H425 V488"/><path class="arrow" d="M775 446 V470 H615 V488"/>
<rect x="265" y="493" width="510" height="65" rx="8" fill="#172e38"/><text x="520" y="519" text-anchor="middle" class="title" style="fill:white">③ 0.5 × 神经预测 ＋ 0.5 × 核预测</text><text x="520" y="545" text-anchor="middle" class="note" style="fill:#d5e1e7">在原表型尺度融合；权重在外层评价前固定</text>
<path class="arrow" d="M520 558 V580"/><text x="520" y="606" text-anchor="middle" class="title">四环境产量预测 · 折均宏 PCC 0.526506</text>
</svg></div><p id="diagram-explanation" class="diagram-explanation" role="status">两条分支分别完成内层选模与完整开发集重拟合，再在预测空间固定等权平均。这里没有联合训练的新融合头，也不根据外层结果改权重。</p>
<figcaption><b>图 1 · 最优候选的完整推理结构。</b>n = 599，4 个环境；图中表示结构与预测聚合，无误差条。来源：quantile_tabm.py、quantile_tabm_final.py、robust_icm.py 及 2026-11-06 最终汇总。环境编号对应原表型列 1、2、4、5。</figcaption></figure>'''


def build():
    qpath = ROOT/'results/quantile_tabm_final_20261106/summary.json'
    mpath = ROOT/'results/mulphe_unified_parallel_20261112/summary.json'
    q = json.loads(qpath.read_text())
    m = json.loads(mpath.read_text())
    systems = [(name, q['macro'][key], q['per_environment'][key]) for key, name in [
        ('quantile_huber','固定等权分位数–Huber 融合'), ('huber','Huber ICM'), ('quantile','四独立分位数 TabM')]]
    for key, value in m['systems'].items():
        selection, arm = key.split('/')
        name = ('三内折选配置' if selection == 'selected' else '公开默认配置') + ' Mul-PheG2P · ' + {
            'scalar_with_target':'标量融合', 'hidden_with_target':'隐藏融合（校准）',
            'hidden_with_target_uncalibrated':'隐藏融合（未校准）'}[arm]
        systems.append((name,value['macro'],value['environment']))
    rows = ''.join('<tr><th scope="row">'+esc(name)+'</th>'+''.join(f'<td>{values[k]:.6f}</td>' for k in ['pcc','rmse','mae'])+
                   ''.join(f'<td>{env[e]["pcc"]:.6f}</td>' for e in ['1','2','4','5'])+'</tr>' for name,values,env in systems)
    bars = ''.join(f'<div class="barrow"><span>{esc(name)}</span><div class="track"><div class="bar" style="width:{values["pcc"]*100:.6f}%"></div></div><strong>{values["pcc"]:.6f}</strong></div>' for name,values,_ in systems)
    head = '''<header><nav><a href="../README.md">项目说明</a><a href="../ref/index.html">参考文献</a><a href="../results/index.html">运行目录</a><a href="#comparison">模型总表</a><a href="#best">最优候选图解</a><a href="#history">历次结果</a><a href="#architectures">架构库</a></nav>
<p class="kicker">WHEAT599 / RESEARCH ATLAS / 2026-11-13</p><h1>Wheat599 · 模型结果与架构</h1><p class="lead">现役候选：独立分位数 TabM 与多环境 Huber ICM 的固定等权融合。</p><p>599 份材料 · 1,279 个二值 DArT 标记 · 4 个已知环境。预测完全留出的新材料；输入仅为标记，未使用谱系矩阵。</p></header><main id="main">'''
    body = head + result_workbench(systems) + '''<details class="legacy-overview"><summary>已核对统一协议：完整指标与来源</summary><section aria-labelledby="overview"><h2 id="overview">统一协议下的结果</h2><div class="grid">
<article class="card best"><span class="tag">当前固定融合 · 折均宏 PCC</span><div class="big">0.526506</div><p>相对 Huber ICM 增加 0.004761；是项目内描述性差值。</p></article>
<article class="card"><span class="tag">训练与评价</span><h3>10 外折 × 3 内折</h3><p>神经模型使用种子 42、2026，先平均预测再评分。所有预处理和选择限于拟合集及内层验证集。</p></article>
<article class="card"><span class="tag">最新完成 · Mul-PheG2P</span><h3>0.438225 / 0.443596</h3><p>分别为调优与默认标量融合。1,920 开发单元、320 外层重拟合，0 运行失败。</p></article></div>
<p>主指标是 40 个环境–外折 PCC 等权平均；折均与合并 OOF 不同。外层划分已多轮用于研究，不是新的独立验证，不计算独立折 p 值或置信区间，也不根据本页重选融合权重。不同预算的完整系统比较不单独归因于架构。</p>
<figure>''' + bars + '''<figcaption><b>图 2 · 固定融合的折均宏 PCC 高于本页已核对的强参照及完整 Mul-PheG2P 适配。</b>n = 599，四环境、十外折；横条从 0 起、满宽为 PCC = 1，无误差条。来源：2026-11-06 分位数最终汇总与 2026-11-13 Mul-PheG2P 汇总；不是文献成绩排名。确切值见下表。</figcaption></figure>
<div class="table-wrapper" tabindex="0" role="region" aria-label="统一协议指标"><table><caption>折均指标；Env1–4 对应原表型列 1、2、4、5</caption><thead><tr><th scope="col">完整系统</th><th scope="col">PCC ↑</th><th scope="col">RMSE ↓</th><th scope="col">MAE ↓</th><th scope="col">Env1 PCC</th><th scope="col">Env2 PCC</th><th scope="col">Env3 PCC</th><th scope="col">Env4 PCC</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p>来源：<a href="../results/quantile_tabm_final_20261106/summary.json">分位数与参照 JSON</a> · <a href="../results/mulphe_unified_parallel_20261112/summary.json">Mul-PheG2P JSON</a> · <a href="EVALUATION_PROTOCOL.md">统一协议与搜索预算</a></p></section></details>'''
    body += '''<section id="best"><h2>最优候选是什么，如何工作？</h2>
<p>这是两个已经独立训练的预测器的平均，不是一张端到端联合网络。神经分支为每个环境学习单独的非线性标记函数；统计分支通过材料相似性和四环境相关性联合预测。融合在原表型尺度进行，权重在外层评价前固定为各 50%。</p>
''' + candidate_diagram() + '''
<details><summary>展开训练规则、损失公式与实现证据</summary>
<h3>A：为什么叫“分位数 TabM”？</h3>
<p>TabM 是参数高效的 MLP 集成。成员共享大部分矩阵参数，并用成员特异的乘性参数产生不同预测；成员数不是独立训练种子数。本项目全标记 raw 分支不使用分组 tokenizer、Transformer 注意力或预训练权重。四环境分别训练，避免把共享环境模型误称为当前候选。</p>
<p>每个成员输出从 0.001 到 0.999 均匀分布的 99 个分位点。pinball 损失对低估与高估施加不同权重：分位点越高，低估的代价越大。损失先对 99 个点求和，再对材料和成员平均。预测时将所有分位点和成员平均为一个数，因此本项目报告的是点预测精度；未验证这些分位点为校准的预测区间，也未强制输出单调。</p>
<figure><math display="block" aria-label="pinball 损失等于误差乘以分位点减去误差小于零的指示函数"><msub><mi>ρ</mi><mi>τ</mi></msub><mo>(</mo><mi>u</mi><mo>)</mo><mo>=</mo><mi>u</mi><mo>(</mo><mi>τ</mi><mo>−</mo><msub><mn>1</mn><mrow><mi>u</mi><mo>&lt;</mo><mn>0</mn></mrow></msub><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>u</mi><mo>=</mo><mi>y</mi><mo>−</mo><msub><mover><mi>q</mi><mo>^</mo></mover><mi>τ</mi></msub></math><figcaption><b>式 1 · 分位数损失。</b>u 为标准化表型与对应分位点预测之差，τ 为分位点水平；与官方分位数回归代码的损失定义对应。</figcaption></figure>
<p>八组预定配置覆盖 2–4 层、宽度 128/256、16/32 成员以及绑定的优化与正则设置。按每环境三内折的双种子平均预测 PCC 选配置，再按各自种子的三个最佳 epoch 中位数重拟合完整开发集。80 次外层拟合、0 失败；轮数 10–103、中位数 21。不能把内部成员解释为 16/32 次独立重训练。</p>
<h3>B：Huber ICM 如何共享四个环境的信息？</h3>
<p>材料核 K 由加性核和 RBF 核按折内选定比例组合；前者比较中心化标记的线性相似性，后者按标记距离给出平滑非线性相似性。环境矩阵 B 是单位矩阵与拟合表型相关矩阵的收缩组合。ICM（内生共区域化模型）让同一材料相似性结构通过 B 在环境间共享；并不直接读入测试材料的任何表型。</p>
<figure><math display="block" aria-label="联合核等于环境矩阵与材料核的克罗内克积"><mi>Ω</mi><mo>=</mo><mi>B</mi><mo>⊗</mo><mi>K</mi><mo>,</mo><mspace width="1em"/><mi>B</mi><mo>=</mo><mo>(</mo><mn>1</mn><mo>−</mo><mi>η</mi><mo>)</mo><mi>I</mi><mo>+</mo><mi>η</mi><msub><mi>C</mi><mi>Y</mi></msub></math><figcaption><b>式 2 · 材料与环境的可分离核结构。</b>C 为仅由拟合材料表型估计的四环境相关矩阵；η 控制共享程度，η = 0 时环境核为单位阵。Ω 表示信号核，不含观测噪声项。</figcaption></figure>
<p>实现先将材料核分解为特征，将 B 分解为环境因子，再求带截距和岭惩罚的凸 Huber 回归。小残差用平方损失，大残差转为线性增长，限制其梯度影响而不删除材料。阈值由拟合集初始平方损失解的残差 MAD 与折内选定倍数组成，使用 L-BFGS-B 求解；这是惩罚风险最小化，不是完整贝叶斯 Huber GP 后验采样。基础核/正则 420 配置与稳健阶段 24 配置依次选择，不是一次联合网格。</p>
<h3>融合的证据与解释边界</h3>
<p>融合 PCC 0.526506，分位数 TabM 为 0.518829，Huber 为 0.521745；RMSE / MAE 同时见主表。平均两种不同函数可以改变误差方向与尺度，本次完整系统获得小幅提升；这些数字本身不能证明某个生物学互作、稳健损失机制或跨数据集领先。主候选仍是固定系统，不把各环境外层最优配置拼成新模型。</p>
<p>实现：<a href="../src/wheat599/quantile_tabm.py">分位数训练</a> · <a href="../src/wheat599/genomic_tabm.py">TabM 适配</a> · <a href="../src/wheat599/robust_icm.py">Huber ICM</a> · <a href="../configs/environment_tabm.yaml">八配置</a> · <a href="QUANTILE_TABM_FINAL.md">重拟合协议</a>。文献：<a href="https://arxiv.org/abs/2410.24210">TabM</a>、<a href="https://github.com/yandex-research/tabular-regression">官方分位数回归</a>；全部参考资料见 <a href="../ref/index.html">文献讲解</a>。</p></details></section>'''
    # Render every result section, including failures and interruptions, without numeric reinterpretation.
    result_text = (ROOT/'docs/RESULTS.md').read_text()
    sections = re.split(r'^## ', result_text, flags=re.M)[1:]
    body += '<details class="archive"><summary>完整运行记录与停止说明</summary><section id="history"><h2>历次运行结果与停止记录</h2><p>完整保留 RESULTS 各节的评价口径、负结果及运行状态；开发、模拟、其他数据集和无效记录不会进入上方统一比较。检索标题或正文，然后展开记录。</p><div class="controls"><label>检索记录 <input type="search" data-filter=".history-item" data-count="history-count" placeholder="例如：蒸馏、失败、Wheat2000"></label><span id="history-count" role="status">'+str(len(sections))+' 项</span></div>'
    for i, section in enumerate(sections):
        title, _, content = section.partition('\n')
        body += f'<details class="history-item" id="run-{i}"><summary>{esc(title)}</summary><div>{markdown(content)}</div></details>'
    body += '</section></details><details class="archive"><summary>架构库与完整模型方案</summary><section id="architectures"><h2>历次模型的架构与实际配方</h2><p>以下展开页来自各模型原方案，包含输入、模块、损失、选择和适配差异。历史方案中的运行指令不表示当前授权；完成状态以本页结果记录为准。早期模型和统计核模型见研究路线与 BH-LMC；当前最优系统见上方完整讲解。</p><div class="controls"><label>检索架构 <input type="search" data-filter=".architecture-item" data-count="arch-count" placeholder="例如：TabM、残差、ICM"></label><span id="arch-count" role="status"></span></div>'
    body += markdown("""
### 早期基线与架构家族速读

| 系统 | 从输入到预测的实际路径 | 关键区别与实现 |
|---|---|---|
| RR-BLUP / 加性核岭 | 拟合集中心化的标记 → 带 L2 惩罚的线性效应 → 产量 | 正则在训练/验证内选择；[核实现](../src/wheat599/kernel_baselines.py) |
| Gaussian / 加性＋RBF 核岭 | 标记距离 → Gaussian 核，或与加性核混合 → 带截距核岭预测 | 非线性来自距离核，不等同证明生物学上位性；[实现](../src/wheat599/kernel_baselines.py) |
| A 架构 FT-Transformer | 标记逐列编码 → Transformer 交互 → 回归输出 | 历史配置宽度 64、2 层、4 头，单种子 42；[配置](../configs/bimsgp.yaml)。A 是架构标签，不是谱系 |
| RR＋A 残差修正 | 固定 RR 预测＋FT 网络预测残差 | 输出层零初始化，验证 PCC 选择 RR-only 或修正分支；[实现](../scripts/run_residual.py) |
| Direct FT-T | 每标记仿射 token → 一层全标记自注意力 → 位点相关的低秩线性读出 | 没有 query-key 池化；[实现](../src/wheat599/direct_ftt.py) |
| 诱导注意力 FT-T | 共享低秩 tokenizer → 标记与诱导槽双向注意力 → 多查询池化＋MLP | 诱导槽不解释为基因通路；[实现](../src/wheat599/induced_ftt.py) |
| MoEGP | 标记 → 门控选择 MLP 专家 → 加权专家输出 | 独立专家与上游共享专家引用两种完整系统分开记录；[适配入口](../scripts/run_moegp.py)、[对照说明](WHEATGP_MOEGP_FOLLOWUP.md) |
| WheatGP | 五段标记 → 公开 CNN–LSTM 网络 → 回归 | 对实际卷积输出长度适配；原标记/PCA 250 在内层选择；[说明](WHEATGP.md) |
| BiMSGP（历史运行） | 单标记线性嵌入 → 正/反向 Mamba 残差块 → 对齐后拼接 → 展平与 MLP 解码 | 固定源码 71084c8，内层比较宽度/状态 32 或 64；[原配置](../configs/bimsgp_nested.yaml)、[适配入口](../scripts/run_bimsgp_nested.py)。本轮不运行 |
| DNAwhisper | 卷积标记嵌入 → GFI 编码/交叉注意力解码与专家、池化 → 四表型输出 | 拟合集内监督预训练后微调，不能称无标签预训练；[配置与适配差异](../configs/dnawhisper_nested.yaml)、[适配实现](../src/wheat599/dnawhisper/adapter.py) |
| RAGP | 每环境编码器 → 检索五个拟合材料邻居 → 编码均值与标签均值融合 → 残差回归 | PCA95 仅拟合集拟合，参考库按 epoch 更新，按材料 ID 排除自检索；[实现及与上游差异](../src/wheat599/ragp.py) |
| 检索 TabM | TabM 第一层成员表示 → 成员特异的参考材料聚合 → 后续 TabM 层 | 参考键与查询共用当前表示，参考标签只来自拟合集；[说明](RETRIEVAL_TABM.md) |
| TabICLv2 | 已预训练表格模型＋拟合材料作为上下文 → 查询材料回归 | 官方权重、通用合成任务续训、遗传合成任务续训分开比较，仅开发；[说明](TABICL_GENOMIC.md) |
| Laplace RFM | 拉普拉斯核回归 → 梯度外积更新完整特征矩阵 → 再拟合核预测器 | 四环境独立，1,279×1,279 度量矩阵，9 配置与 0–5 次更新在内层选；[说明](RFM_DEVELOPMENT.md) |
| ICM / BH-LMC / GET-ICM | 材料核与环境协方差结构 → 正则化多环境核预测 | ICM 共享一个材料核；BH-LMC 为不同核设环境结构；GET 添加结构化残差。各配方见下方原方案 |
| MSE / 分位数 TabM | 全标记 → 参数高效 MLP 集成 → 单值或 99 分位点输出 | 共享、独立、条件化环境分支及不同损失分别记账；最优候选使用四独立分位数分支 |
| Mul-PheG2P | 源环境与目标环境编码器预训练 → 目标适配 → 注意力融合 | 标量预测融合与隐藏表示融合分列；有/无折内校准分列；[完整适配](MULPHE_UNIFIED.md) |
| 模拟推断与后续变体 | 材料—位点推断、关系约束、交互、检索、蒸馏、条件网络或稀疏树 | 每个具体路径见下方独立说明与历史结果；模拟收益不作为 Wheat599 证据 |
""")

    excluded = {'RESULTS.md','MODEL_COMPARISON.md','REFERENCES.md','LICENSE_NOTES.md','EVALUATION_PROTOCOL.md','DATA_FACTS.md','G0_VERSION.md','TRIGGER_TEMPLATE.md','REVIEW_PROTOCOL.md','MIGRATION.md','HANDOVER_20261029.md','HANDOVER_20261030.md'}
    for path in sorted((ROOT/'docs').glob('*.md')):
        if path.name in excluded or path.name.startswith('面向 '):
            continue
        content = path.read_text()
        title = next((s.lstrip('# ') for s in content.splitlines() if s.startswith('# ')), path.stem)
        body += f'<details class="architecture-item"><summary>{esc(title)}</summary><div><p><a href="{esc(path.name)}">原始说明</a></p>{markdown(content)}</div></details>'
    body += '</section></details><section><h2>证据版本</h2><p>可视化不重新评分，不重新训练。主图从 JSON 直接取数；历史全文与方案从 Markdown 生成。下列 SHA-256 指向本次生成所用内容。</p><ul>'
    for path in [qpath,mpath,ROOT/'docs/RESULTS.md']:
        body += f'<li>{esc(path.relative_to(ROOT))}<br><code>{hashlib.sha256(path.read_bytes()).hexdigest()}</code></li>'
    body += '</ul></section></main>'
    page('Wheat599 · 模型结果与架构',body,ROOT/'docs/models.html')
    # Inventory is a navigation view, not an inferred completion verdict.
    inventory = '<header><nav><a href="../docs/models.html">结果与架构</a><a href="../ref/index.html">参考文献</a><a href="../README.md">项目说明</a></nav><p class="kicker">WHEAT599 / ARTIFACTS</p><h1>运行目录与证据文件</h1><p>保持 results/&lt;run&gt;/ 原路径，避免改动清单、配置与哈希引用。目录存在不等于运行完成；状态按原记录阅读。冒烟、无效及中断条目仅供追溯，不用于性能排名。</p></header><main id="main"><div class="controls"><label>检索目录 <input type="search" data-filter=".run-folder" data-count="folder-count"></label><span id="folder-count" role="status"></span></div>'
    for directory in sorted((ROOT/'results').iterdir()):
        if not directory.is_dir() or directory.name.startswith('.'):
            continue
        filenames = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ('.json','.md','.yaml','.txt','.csv') and not re.search(r'oof|prediction|trajectory|trajector|validation.csv',p.name)]
        links = ' · '.join(f'<a href="{esc(p.relative_to(ROOT/"results"))}">{esc(p.name)}</a>' for p in filenames)
        inventory += f'<details class="run-folder"><summary>{esc(directory.name)}</summary><div>{links or "本地目录；无顶层小型说明文件"}</div></details>'
    inventory += '</main>'
    page('Wheat599 · 运行目录',inventory,ROOT/'results/index.html')
    print(f'Built docs/models.html ({len(sections)} result sections) and results/index.html')


if __name__ == '__main__':
    build()

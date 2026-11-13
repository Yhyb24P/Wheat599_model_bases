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
    head = '''<header><nav><a href="../README.md">项目说明</a><a href="../ref/index.html">参考文献</a><a href="../results/index.html">运行目录</a><a href="#best">最优候选原理</a><a href="#history">历次结果</a><a href="#architectures">架构库</a></nav>
<p class="kicker">WHEAT599 / RESEARCH ATLAS / 2026-11-13</p><h1>从基因型到四环境表型</h1><p class="lead">现役候选：独立分位数 TabM 与多环境 Huber ICM 的固定等权融合。</p><p>599 份材料 · 1,279 个二值 DArT 标记 · 4 个已知环境。预测完全留出的新材料；输入仅为标记，未使用谱系矩阵。</p></header><main id="main">'''
    body = head + '''<section aria-labelledby="overview"><h2 id="overview">统一协议下的结果</h2><div class="grid">
<article class="card best"><span class="tag">当前固定融合 · 折均宏 PCC</span><div class="big">0.526506</div><p>相对 Huber ICM 增加 0.004761；是项目内描述性差值。</p></article>
<article class="card"><span class="tag">训练与评价</span><h3>10 外折 × 3 内折</h3><p>神经模型使用种子 42、2026，先平均预测再评分。所有预处理和选择限于拟合集及内层验证集。</p></article>
<article class="card"><span class="tag">最新完成 · Mul-PheG2P</span><h3>0.438225 / 0.443596</h3><p>分别为调优与默认标量融合。1,920 开发单元、320 外层重拟合，0 运行失败。</p></article></div>
<p>主指标是 40 个环境–外折 PCC 等权平均；折均与合并 OOF 不同。外层划分已多轮用于研究，不是新的独立验证，不计算独立折 p 值或置信区间，也不根据本页重选融合权重。不同预算的完整系统比较不单独归因于架构。</p>
<figure>''' + bars + '''<figcaption><b>图 1 · 固定融合的折均宏 PCC 高于本页已核对的强参照及完整 Mul-PheG2P 适配。</b>n = 599，四环境、十外折；横条从 0 起、满宽为 PCC = 1，无误差条。来源：2026-11-06 分位数最终汇总与 2026-11-13 Mul-PheG2P 汇总；不是文献成绩排名。确切值见下表。</figcaption></figure>
<div class="table-wrapper" tabindex="0" role="region" aria-label="统一协议指标"><table><caption>折均指标；Env1–4 对应原表型列 1、2、4、5</caption><thead><tr><th scope="col">完整系统</th><th scope="col">PCC ↑</th><th scope="col">RMSE ↓</th><th scope="col">MAE ↓</th><th scope="col">Env1 PCC</th><th scope="col">Env2 PCC</th><th scope="col">Env3 PCC</th><th scope="col">Env4 PCC</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p>来源：<a href="../results/quantile_tabm_final_20261106/summary.json">分位数与参照 JSON</a> · <a href="../results/mulphe_unified_parallel_20261112/summary.json">Mul-PheG2P JSON</a> · <a href="EVALUATION_PROTOCOL.md">统一协议与搜索预算</a></p></section>'''
    body += '''<section id="best"><h2>最优候选是什么，如何工作？</h2>
<p>这是两个已经独立训练的预测器的平均，不是一张端到端联合网络。神经分支为每个环境学习单独的非线性标记函数；统计分支通过材料相似性和四环境相关性联合预测。融合在原表型尺度进行，权重在外层评价前固定为各 50%。</p>
<figure><div class="node">输入：一份新材料的 1,279 个 0/1 标记；沿用该外折拟合集的预处理统计量。</div><div class="diagram"><div class="node"><b>A · 四独立分位数 TabM</b><p>每环境一套模型 → 共享主权重的多个 MLP 成员 → 每成员 99 个分位点 → 成员与分位点平均 → 反标准化 → 双种子预测平均。</p></div><div class="node"><b>B · 多环境 Huber ICM</b><p>加性核＋RBF 材料核 → 与环境相关矩阵结合 → Huber 损失与岭惩罚拟合 → 四环境联合预测 → 反标准化。</p></div></div><div class="merge">0.5 × A + 0.5 × B → 四环境产量预测</div><figcaption><b>图 2 · 当前候选的完整推理路径。</b>结构示意，无误差条；数据任务 n = 599，四环境。实现来源：quantile_tabm.py、quantile_tabm_final.py、robust_icm.py，2026-11-13 核对。两分支不在测试集上学习融合权重。</figcaption></figure>
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
<p>实现：<a href="../src/wheat599/quantile_tabm.py">分位数训练</a> · <a href="../src/wheat599/genomic_tabm.py">TabM 适配</a> · <a href="../src/wheat599/robust_icm.py">Huber ICM</a> · <a href="../configs/environment_tabm.yaml">八配置</a> · <a href="QUANTILE_TABM_FINAL.md">重拟合协议</a>。文献：<a href="https://arxiv.org/abs/2410.24210">TabM</a>、<a href="https://github.com/yandex-research/tabular-regression">官方分位数回归</a>；全部参考资料见 <a href="../ref/index.html">文献讲解</a>。</p></section>'''
    comparison = (ROOT/'docs/MODEL_COMPARISON.md').read_text().split('## 全模型总对比表', 1)[1].split('\n## ', 1)[0]
    grouped = {}
    for line in comparison.splitlines():
        if not line.startswith('| ') or line.startswith('| 评价层'):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) != 6:
            raise ValueError(f'Unexpected model comparison row: {line}')
        grouped.setdefault(cells[0], []).append(cells[1:])
    body += '<section id="comparison"><h2>历史模型指标浏览</h2><p>按原评价层分组，不把开发分数与外层分数混排。同组内仍可能有预算、种子或历史协议差异，来源逐行保留；重复参照沿用原表，不推断为独立复验。PCC 横条仅辅助读数，0–1 固定尺度，未定义值保留文字。</p>'
    for group, records in grouped.items():
        body += f'<details><summary>{esc(group)} · {len(records)} 项系统记录</summary><div class="table-wrapper" tabindex="0" role="region" aria-label="{esc(group)}模型指标"><table><caption>{esc(group)} · 原始口径</caption><thead><tr><th scope="col">模型／系统</th><th scope="col">PCC</th><th scope="col">RMSE</th><th scope="col">MAE</th><th scope="col">来源</th></tr></thead><tbody>'
        for name, pcc, rmse, mae, source in records:
            bar = ''
            if re.fullmatch(r'[+-]?\d+\.\d+', pcc):
                value = float(pcc)
                if not -1 <= value <= 1:
                    raise ValueError('PCC outside [-1,1]')
                if value >= 0:
                    bar = f'<div class="track" aria-hidden="true"><div class="bar" style="width:{value*100}%"></div></div>'
            body += f'<tr><th scope="row">{esc(name)}</th><td>{esc(pcc)}{bar}</td><td>{esc(rmse)}</td><td>{esc(mae)}</td><td>{markdown(source)}</td></tr>'
        body += '</tbody></table></div></details>'
    body += '</section>'

    # Render every result section, including failures and interruptions, without numeric reinterpretation.
    result_text = (ROOT/'docs/RESULTS.md').read_text()
    sections = re.split(r'^## ', result_text, flags=re.M)[1:]
    body += '<section id="history"><h2>历次运行结果与停止记录</h2><p>完整保留 RESULTS 各节的评价口径、负结果及运行状态；开发、模拟、其他数据集和无效记录不会进入上方统一比较。检索标题或正文，然后展开记录。</p><div class="controls"><label>检索记录 <input type="search" data-filter=".history-item" data-count="history-count" placeholder="例如：蒸馏、失败、Wheat2000"></label><span id="history-count" role="status">'+str(len(sections))+' 项</span></div>'
    for i, section in enumerate(sections):
        title, _, content = section.partition('\n')
        body += f'<details class="history-item" id="run-{i}"><summary>{esc(title)}</summary><div>{markdown(content)}</div></details>'
    body += '</section><section id="architectures"><h2>历次模型的架构与实际配方</h2><p>以下展开页来自各模型原方案，包含输入、模块、损失、选择和适配差异。历史方案中的运行指令不表示当前授权；完成状态以本页结果记录为准。早期模型和统计核模型见研究路线与 BH-LMC；当前最优系统见上方完整讲解。</p><div class="controls"><label>检索架构 <input type="search" data-filter=".architecture-item" data-count="arch-count" placeholder="例如：TabM、残差、ICM"></label><span id="arch-count" role="status"></span></div>'
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
    body += '</section><section><h2>证据版本</h2><p>可视化不重新评分，不重新训练。主图从 JSON 直接取数；历史全文与方案从 Markdown 生成。下列 SHA-256 指向本次生成所用内容。</p><ul>'
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

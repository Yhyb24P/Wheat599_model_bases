"""The offline report must preserve recorded values and all historical sections."""
import importlib.util
import json
import re
import shutil
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.history = 0
        self.math = 0

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        self.history += tag == 'details' and attrs.get('class') == 'history-item'
        self.math += tag == 'math'


class ReportTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('pandoc'), 'Report build requires pandoc')
    def test_report_preserves_sources_and_is_rebuildable(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check_report(Path(directory))

    @unittest.skipUnless(shutil.which('bibtex'), 'BibTeX parser is not installed')
    def test_reference_export_is_valid_bibtex(self):
        spec = importlib.util.spec_from_file_location('references', ROOT/'ref/build.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        keys = [r['key'] for r in module.REFERENCES]
        self.assertEqual(len(keys), len(set(keys)))
        page = module.index_page()
        self.assertEqual(page.count('class="refcard"'), len(keys))
        for reference in module.REFERENCES:
            self.assertTrue(reference['brief'].strip(), reference['key'])
            self.assertTrue(reference['relation'].strip(), reference['key'])
            self.assertIn('id="' + reference['key'] + '"', page)
            self.assertIn(module.rich(reference['brief']), page)
            self.assertIn(module.rich(reference['verification']), page)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary/'references.bib').write_text('\n\n'.join(module.bibtex(r) for r in module.REFERENCES))
            (temporary/'check.aux').write_text(
                r'\relax' + '\n' + r'\citation{*}' + '\n' +
                r'\bibdata{references}' + '\n' + r'\bibstyle{plain}' + '\n')
            result = subprocess.run(['bibtex', 'check.aux'], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            bibliography = (temporary/'check.bbl').read_text()
            self.assertEqual(bibliography.count(r'\bibitem{'), len(keys))

    def check_report(self, tmp_path):
        spec = importlib.util.spec_from_file_location('report', ROOT/'scripts/build_report.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        # Build against a copy so the check cannot replace the checked-in reports.
        for folder in ['docs', 'results']:
            (tmp_path/folder).mkdir()
        for path in (ROOT/'docs').glob('*.md'):
            shutil.copyfile(path, tmp_path/'docs'/path.name)
        for name in ['quantile_tabm_final_20261106', 'mulphe_unified_parallel_20261112']:
            (tmp_path/'results'/name).mkdir()
            shutil.copyfile(ROOT/'results'/name/'summary.json', tmp_path/'results'/name/'summary.json')
        for name in ['results/quantile_tabm_final_manifest_20261106/manifest.json',
                     'results/environment_tabm_development_20261105/provenance.json']:
            target = tmp_path/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        report.ROOT = tmp_path
        report.build()
        output = (tmp_path/'docs/models.html').read_text()
        document = Document()
        document.feed(output)
        expected = len(re.findall(r'^## ', (ROOT/'docs/RESULTS.md').read_text(), re.M))
        assert document.history == expected
        assert not [key for key, count in Counter(document.ids).items() if count > 1]
        assert document.math >= 5
        mechanisms = json.loads(re.search(r'<script id="mechanism-data" type="application/json">(.*?)</script>', output, re.S)[1])
        self.assertEqual(len(mechanisms), 40)
        frozen = json.loads((ROOT/'results/quantile_tabm_final_manifest_20261106/manifest.json').read_text())
        for model in mechanisms:
            source = [r for r in frozen['rows'] if r['fold'] == model['fold'] and r['environment_index'] == model['environment']]
            self.assertEqual({r['config_index'] for r in source}, {model['config_index']})
            self.assertEqual(model['epochs'], {str(r['seed']):r['epochs'] for r in source})
        self.assertEqual((mechanisms[0]['n_blocks'], mechanisms[0]['d_block'], mechanisms[0]['k']), (4,128,32))
        self.assertIn('LinearBatchEnsemble', output)
        self.assertIn('最后输出头不共享', output)
        q = json.loads((ROOT/'results/quantile_tabm_final_20261106/summary.json').read_text())
        for key, label in [('quantile_huber', '固定等权分位数–Huber 融合'), ('quantile', '四独立分位数 TabM'), ('huber', 'Huber ICM')]:
            row = '<tr><th scope="row">' + label + '</th>'
            row += ''.join(f'<td>{q["macro"][key][metric]:.6f}</td>' for metric in ['pcc', 'rmse', 'mae'])
            row += ''.join(f'<td>{q["per_environment"][key][env]["pcc"]:.6f}</td>' for env in ['1', '2', '4', '5'])
            assert row + '</tr>' in output
            for metric in ['pcc', 'rmse', 'mae']:
                assert f'{q["macro"][key][metric]:.6f}' in output
        m = json.loads((ROOT/'results/mulphe_unified_parallel_20261112/summary.json').read_text())
        for system in m['systems'].values():
            for metric in ['pcc', 'rmse', 'mae']:
                assert f'{system["macro"][metric]:.6f}' in output
        payload = json.loads(re.search(r'<script id="model-data" type="application/json">(.*?)</script>', output, re.S)[1])
        records = payload['records']
        comparison = (ROOT/'docs/MODEL_COMPARISON.md').read_text().split('## 全模型总对比表', 1)[1].split('\n## ', 1)[0]
        original = [tuple(cell.strip() for cell in line.strip('|').split('|'))
                    for line in comparison.splitlines() if line.startswith('| ') and not line.startswith('| 评价层')]
        self.assertEqual(len(records), len(original) + 9)
        actual = {(r['stage'], r['name'], r['pcc'], r['rmse'], r['mae']) for r in records}
        for stage, name, pcc, rmse, mae, _ in original:
            self.assertIn((stage, name.replace('`', ''), pcc, rmse, mae), actual)
        self.assertEqual(records[0]['role'], '主候选')
        first = {r['family'] for r in records[:11]}
        self.assertTrue({'WheatGP', 'MoEGP', 'BiMSGP', 'DNAwhisper', 'RAGP', 'TabICLv2'} <= first)
        self.assertIn('RAGP', next(r['name'] for r in records if r['family'] == 'RAGP'))
        # A comparison's title must not turn its TabM control into the RAGP architecture.
        self.assertEqual(report.model_family('单成员检索网络', 'RAGP 同协议参照'), 'TabM')
        undefined = [r for r in records if '无定义' in r['pcc']]
        self.assertTrue(undefined)
        self.assertTrue(all(r['values']['pcc'] is None for r in undefined))
        self.assertTrue(all(r['family'] in payload['architectures'] for r in records))
        self.assertIn('id="candidate-svg-title"', output)
        if shutil.which('node'):
            script = tmp_path/'report.js'
            script.write_text(report.JS)
            result = subprocess.run(['node', '--check', str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        assert '无定义（5/40 折）' in output
        assert '中点分位数批次按研究范围调整停止' in output
        assert '无效结果' in output
        assert 'https://cdn' not in output
        assert (tmp_path/'results/index.html').exists()

if __name__ == '__main__':
    unittest.main()

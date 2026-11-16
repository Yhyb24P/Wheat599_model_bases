"""SoyBase loader self-check on synthetic HapMap + GRIN observations."""
import gzip
import json
import unittest
from pathlib import Path
import tempfile

import numpy as np

from genopred.data import soybase


class SoyBaseLoaderTest(unittest.TestCase):
    def test_norm_pi(self):
        self.assertEqual(soybase._norm_pi('PI 253651 A'), 'PI253651A')
        self.assertEqual(soybase._norm_pi('pi86046'), 'PI86046')

    def test_hapmap_dosage_and_monomorphic_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.hmp.gz'
            header = ['rs#', 'alleles', 'chrom', 'pos', 'strand', 'assembly#',
                      'center', 'protLSID', 'assayLSID', 'panelLSID', 'QCcode',
                      'PI1', 'PI2', 'PI3']
            rows = [
                ['rs1', 'A/G', 'Gm01', '1', '+', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'A', 'G', 'R'],
                ['rs2', 'C/T', 'Gm01', '2', '+', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'C', 'Y', 'N'],
                ['rs3', 'A', 'Gm01', '3', '+', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'A', 'A', 'A'],
            ]
            with gzip.open(path, 'wt') as handle:
                handle.write('\t'.join(header) + '\n')
                for row in rows:
                    handle.write('\t'.join(row) + '\n')
            ids, snps, X = soybase.parse_hapmap(path)
            self.assertEqual(ids, ['PI1', 'PI2', 'PI3'])
            self.assertEqual(snps, ['rs1', 'rs2'])  # rs3 monomorphic dropped
            np.testing.assert_array_equal(X[0], [0.0, 0.0])   # PI1 A=0, C=0
            np.testing.assert_array_equal(X[1], [2.0, 1.0])   # PI2 G=2, Y(het C/T)=1
            self.assertEqual(X[2, 0], 1.0)                    # PI3 R(het A/G)=1
            self.assertTrue(np.isnan(X[2, 1]))                # PI3 N missing

    def test_phenotype_filter_and_conflict_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = Path(tmp) / 'obs.json'
            records = [
                {'cultivar_name': 'PI 1', 'trait_name': 'Flower color', 'value': 'W'},
                {'cultivar_name': 'PI 2', 'trait_name': 'Flower color', 'value': 'P'},
                {'cultivar_name': 'PI 3', 'trait_name': 'Flower color', 'value': 'W'},
                {'cultivar_name': 'PI 3', 'trait_name': 'Flower color', 'value': 'P'},  # conflict
                {'cultivar_name': 'PI 1', 'trait_name': 'Flower color', 'value': 'Dp'},  # non-benchmark
                {'cultivar_name': 'PI 1', 'trait_name': 'Stem termination type', 'value': 'N'},
                {'cultivar_name': 'PI 2', 'trait_name': 'Stem termination type', 'value': 'S'},
                {'cultivar_name': 'PI 3', 'trait_name': 'Stem termination type', 'value': 'D'},
                {'cultivar_name': 'FC 999', 'trait_name': 'Flower color', 'value': 'W'},  # unmatched
            ]
            obs.write_text(json.dumps(records))
            Y, stats = soybase.parse_phenotype(obs, ['PI1', 'PI2', 'PI3'])
            self.assertEqual(Y[0, 0], 0)   # PI1 FC W
            self.assertEqual(Y[1, 0], 1)   # PI2 FC P
            self.assertEqual(Y[2, 0], -1)  # PI3 FC conflict dropped
            self.assertEqual(Y[0, 3], 0)   # PI1 ST N
            self.assertEqual(Y[1, 3], 1)   # PI2 ST S
            self.assertEqual(Y[2, 3], 2)   # PI3 ST D
            self.assertEqual(stats['FC']['conflict_dropped'], 1)
            self.assertEqual(stats['FC']['labeled_accessions'], 2)


if __name__ == '__main__':
    unittest.main()
"""CPU-only tests: python3 tools/hip/test_summarize_prefill.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from summarize_prefill import parse_lines, summarize

LINE = ('strata prefill timing: 505 tokens, GPU timeline 200 ms, wall 240 ms, '
        'host staging 30 ms: hc read 50 (25.0%) gemm gate/up 100 (50.0%) '
        'wait copy 50 (25.0%)')
HOST = ('strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) '
        '5 ms, waiting for each chunk 220 ms, after each chunk (the draft layer, progress) '
        '8 ms, PLE 3 ms')


class Tests(unittest.TestCase):
    def test_phases_and_host(self):
        rows = parse_lines(['unrelated startup', LINE, 'other diagnostic', HOST])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['tokens'], 505)
        self.assertEqual(rows[0]['phases']['gemm gate/up']['ms'], 100)
        self.assertEqual(rows[0]['host']['ple_ms'], 3)

    def test_weighted_not_mean_of_percentages(self):
        second = LINE.replace('200 ms', '400 ms').replace('100 (50.0%)', '300 (75.0%)')
        result = summarize(parse_lines([LINE, second]))
        top = result['phases'][0]
        self.assertEqual(top['ms'], 400)
        self.assertAlmostEqual(top['percent'], 100 * 400 / 600)

    def test_empty_phases_zero_time(self):
        row = parse_lines(['strata prefill timing: 0 tokens, GPU timeline 0 ms, wall 0 ms, host staging 0 ms:'])
        self.assertEqual(summarize(row)['phases'], [])

    def test_decimal_timing(self):
        row = parse_lines([LINE.replace('200 ms', '200.5 ms')])[0]
        self.assertEqual(row['gpu_ms'], 200.5)

    def test_unknown_phase_syntax_rejected(self):
        with self.assertRaises(ValueError):
            parse_lines([LINE + ' unparsed tail'])

    def test_duplicate_phase_rejected(self):
        with self.assertRaises(ValueError):
            parse_lines([LINE + ' wait copy 0 (0.0%)'])

    def test_malformed_timing_rejected(self):
        with self.assertRaises(ValueError):
            parse_lines(['strata prefill timing: bad'])

    def test_orphan_host_rejected(self):
        with self.assertRaises(ValueError):
            parse_lines([HOST])

    def test_missing_host_is_valid(self):
        self.assertNotIn('host', parse_lines([LINE])[0])

    def test_dispatch_receipt(self):
        row = parse_lines([LINE, HOST,
            'strata prefill dispatch: mmq layers 48, fp16 layers 0, fused layers 0, '
            'group gathers 100, single gathers 20, dequant experts 0, gather bytes 123456, '
            'resident grouped layers 41'])[0]
        self.assertEqual(row['dispatch']['mmq_layers'], 48)
        self.assertEqual(row['dispatch']['routed_grouped_layers'], 41)
        self.assertEqual(row['dispatch']['gather_bytes'], 123456)

    def test_orphan_dispatch_rejected(self):
        with self.assertRaises(ValueError):
            parse_lines(['strata prefill dispatch: mmq layers 48, fp16 layers 0, fused layers 0, '
                'group gathers 100, single gathers 20, dequant experts 0, gather bytes 123456, '
                'resident grouped layers 41'])

    def test_cli_filter_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'engine.log'
            output = Path(tmp) / 'result.json'
            log.write_text(LINE + '\n' + LINE.replace('505 tokens', '4089 tokens') + '\n' + LINE + '\n')
            process = subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_prefill.py')),
                str(log), '--skip', '1', '--tokens', '505', '--output', str(output)],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(output.read_text())['summary']['calls'], 1)

    def test_cli_empty_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'empty.log'
            log.write_text('startup only\n')
            process = subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_prefill.py')),
                str(log)], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('No matching', process.stderr)


if __name__ == '__main__':
    unittest.main()

"""CPU-only replay contract checks; no server, model or GPU."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gyro_probe import replay


class Tests(unittest.TestCase):
    def run_probe(self, *, reused=0, tokens=512, generated=128, duplicate=False):
        with tempfile.TemporaryDirectory() as tmp:
            log, output = Path(tmp) / 'engine.log', Path(tmp) / 'results.json'
            log.write_text('startup\n')
            count = 0
            def fake_response(req, timeout):
                nonlocal count
                count += 1
                body = json.loads(req.data)
                self.assertEqual(body['temperature'], 0)
                self.assertEqual(body['max_tokens'], 128)
                n = 25 if count == 1 else tokens
                line = (f'prompt {n} tokens = {reused} reused + {n-reused} read in 100 ms '
                        f'(5120.0 tok/s), {generated} generated in 1000 ms (128.0 tok/s)\n')
                with log.open('a') as stream:
                    stream.write(line * (2 if duplicate else 1))
                return io.BytesIO(json.dumps({'choices': [{'message': {'role': 'assistant',
                    'content': 'fixture'}, 'finish_reason': 'length'}]}).encode())
            cases = [dict(kind='code', trial=1, expected_prompt_tokens=512,
                          messages=[{'role': 'user', 'content': 'fixed prompt'}])]
            with patch('gyro_probe.urllib.request.urlopen', side_effect=fake_response), patch('builtins.print'):
                rows = replay(cases, 'http://127.0.0.1:18080', 'gyro', log, 'test', output)
            self.assertEqual(json.loads(output.read_text()), rows)
            return rows

    def test_replay(self):
        self.assertEqual(len(self.run_probe()), 2)

    def test_reuse_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.run_probe(reused=1)

    def test_prompt_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'count'):
            self.run_probe(tokens=511)

    def test_output_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Output count'):
            self.run_probe(generated=127)

    def test_multiple_completions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one completed'):
            self.run_probe(duplicate=True)


if __name__ == '__main__':
    unittest.main()

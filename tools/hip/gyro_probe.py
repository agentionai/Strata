#!/usr/bin/env python3
"""Prepare fixed Gyro prompts or replay them against a dedicated local server.

Preparation uses the pack's tokenizer/template (no GPU). Replay checks zero
prefix reuse and actual prompt/output counts. This does not manage services.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TIMING = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms '
                    r'\(([0-9.]+) tok/s\), (\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps',
          'generated', 'decode_ms', 'decode_tps')


def prepare(pack):
    sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
    from serve.server import Service, MockEngine
    from serve.frontend import ChatTemplate
    import strata_tokenizer as st
    folder = pack / 'tokenizer'
    vocab = json.loads((folder / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = st.Tokenizer(tokens, (folder / 'merges.txt').read_text().split('\n'),
                       json.loads((folder / 'token_type.json').read_text()))
    service = Service(MockEngine(tok, 'ready'), tok, ChatTemplate(folder / 'chat_template.jinja'))
    cases = []
    for trial, marker in enumerate(('ALPHA', 'BRAVO', 'CHARLIE'), 1):
        for kind, size in (('code', 512), ('prose', 512), ('json', 512), ('code_long', 4096)):
            if kind.startswith('code'):
                task = 'Write a complete Python module implementing a bounded LRU cache, with detailed docstrings and at least ten executable unit tests. Use this reference material as context:\n'
                context = '\n'.join(f'def reference_{i}(x):\n    return x * {i + 1} if x > {i} else x - {i}\n' for i in range(1500))
            elif kind == 'prose':
                task = 'Write a detailed 1600-word explanation of how a modern water treatment plant handles changing raw water quality, including concrete examples. Reference notes:\n'
                context = ('The plant measures turbidity, temperature, flow, pH and organic carbon. Coagulation, settling, filtration and disinfection have different control timescales. Operators verify sensor readings and account for transport delay. ' * 200)
            else:
                task = 'Return only a JSON array of 80 detailed maintenance tasks, each with id, equipment, description, priority and interval_days. Reference notes:\n'
                context = ('Equipment includes pumps, filters, clarifiers, mixers, chemical dosing systems, flow meters, pH probes, valves and standby generators. Inspection frequency depends on operating hours and failure consequences. ' * 200)
            prefix = f'{marker} {kind} benchmark case {trial}. ' + task
            def count(text):
                return len(service.encode_prompt([{'role': 'user', 'content': text}], None,
                                                 {'enable_thinking': False}))
            lo, hi = 0, len(context)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if count(prefix + context[:mid]) <= size:
                    lo = mid
                else:
                    hi = mid - 1
            candidates = [prefix + context[:cut] for cut in
                          range(max(0, lo - 64), min(len(context), lo + 64) + 1)]
            content = min(candidates, key=lambda text: abs(count(text) - size))
            actual = count(content)
            if abs(actual - size) > 2:
                raise ValueError(f'Expected {size} tokens, got {actual}')
            cases.append(dict(kind=kind, trial=trial, expected_prompt_tokens=actual,
                              messages=[{'role': 'user', 'content': content}]))
    return cases


def replay(cases, url, model, log, label, output):
    results = []
    warmup = dict(kind='warmup', trial=0, messages=[{'role': 'user', 'content':
        'Give a detailed explanation of binary search with Python code and examples.'}])
    for case in [warmup] + cases:
        offset = log.stat().st_size
        body = dict(model=model, messages=case['messages'], max_tokens=128, temperature=0,
                    top_k=1, top_p=1, min_p=0, seed=42, reasoning_effort='none')
        req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=360) as response:
            data = json.load(response)
        deadline = time.monotonic() + 3
        while True:
            with log.open('rb') as stream:
                stream.seek(offset)
                matches = TIMING.findall(stream.read().decode(errors='replace'))
            if matches or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if len(matches) != 1:
            raise ValueError('Expected one completed timing line; use an idle dedicated server and its log')
        metrics = dict(zip(FIELDS, map(float, matches[0])))
        if case['kind'] != 'warmup':
            if metrics['reused'] != 0 or metrics['prompt_tokens'] != case['expected_prompt_tokens']:
                raise ValueError('Prompt reused cached tokens or differs from saved count')
            if metrics['generated'] != 128:
                raise ValueError('Output count differs from the 128-token benchmark cap')
        results.append(dict(label=label, **case, metrics=metrics,
                            wall_s=time.monotonic()-started, response=data))
        output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(dict(label=label, kind=case['kind'], trial=case['trial'], metrics=metrics)), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare')
    prep.add_argument('--pack', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    run = actions.add_parser('run')
    run.add_argument('--requests', type=Path, required=True)
    run.add_argument('--url', default='http://127.0.0.1:18080')
    run.add_argument('--model', default='gyro-s-strata')
    run.add_argument('--engine-log', type=Path, required=True)
    run.add_argument('--label', required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve previous results')
    try:
        if args.action == 'prepare':
            args.output.write_text(json.dumps(prepare(args.pack), indent=2) + '\n')
        else:
            replay(json.loads(args.requests.read_text()), args.url, args.model,
                   args.engine_log, args.label, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

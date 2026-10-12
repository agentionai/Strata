#!/usr/bin/env python3
"""Summarize existing STRATA_PREFILL_TIMING=1 logs; no GPU or model needed.

GPU phases are elapsed compute-stream intervals, including waits, not isolated
kernel runtimes. Instrumented timings are diagnostic, not benchmark results.
Each record is one prefill call; token counts need not equal an API prompt's.
"""
import argparse
import json
from pathlib import Path
import re

NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)'
GPU = re.compile(
    r'^strata prefill timing: (\d+) tokens, GPU timeline (' + NUMBER +
    r') ms, wall (' + NUMBER + r') ms, host staging (' + NUMBER + r') ms:(.*)$')
PHASE = re.compile(r' ([^()]+?) (' + NUMBER + r') \((' + NUMBER + r')%\)')
HOST = re.compile(
    r'^strata prefill timing: host: chunk setup \(PLE rows, the expert stream plan\) (' +
    NUMBER + r') ms, waiting for each chunk (' + NUMBER +
    r') ms, after each chunk \(the draft layer, progress\) (' + NUMBER +
    r') ms, PLE (' + NUMBER + r') ms$')
DISPATCH = re.compile(
    r'^strata prefill dispatch: mmq layers (\d+), fp16 layers (\d+), fused layers (\d+), '
    r'group gathers (\d+), single gathers (\d+), dequant experts (\d+), gather bytes (\d+), '
    r'(?:resident|routed) grouped layers (\d+)$')


def parse_lines(lines):
    records = []
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        dispatch = DISPATCH.fullmatch(line)
        if dispatch:
            if not records or 'dispatch' in records[-1]:
                raise ValueError(f'Line {line_number}: dispatch without a prefill call')
            records[-1]['dispatch'] = dict(zip(
                ('mmq_layers', 'fp16_layers', 'fused_layers', 'group_gathers', 'single_gathers',
                 'dequant_experts', 'gather_bytes', 'routed_grouped_layers'),
                map(int, dispatch.groups())))
            continue
        if line.startswith('strata prefill dispatch:'):
            raise ValueError(f'Line {line_number}: unsupported dispatch line')
        match = GPU.fullmatch(line)
        if match:
            tokens, gpu, wall, staging, suffix = match.groups()
            phases = {}
            consumed = ''
            for phase in PHASE.finditer(suffix):
                name, ms, percent = phase.groups()
                if name in phases:
                    raise ValueError(f'Line {line_number}: duplicate phase {name!r}')
                phases[name] = dict(ms=float(ms), percent=float(percent))
                consumed += phase.group(0)
            if consumed != suffix:
                raise ValueError(f'Line {line_number}: unrecognized phase syntax')
            records.append(dict(line=line_number, tokens=int(tokens), gpu_ms=float(gpu),
                                wall_ms=float(wall), host_staging_ms=float(staging),
                                phases=phases))
            continue
        match = HOST.fullmatch(line)
        if match:
            if not records or 'host' in records[-1]:
                raise ValueError(f'Line {line_number}: host timing without a prefill call')
            records[-1]['host'] = dict(zip(
                ('setup_ms', 'wait_ms', 'after_chunk_ms', 'ple_ms'),
                map(float, match.groups())))
        elif line.startswith('strata prefill timing:'):
            raise ValueError(f'Line {line_number}: unsupported timing line')
    return records


def summarize(records):
    total = sum(row['gpu_ms'] for row in records)
    phases = {}
    for row in records:
        for name, values in row['phases'].items():
            phases[name] = phases.get(name, 0.0) + values['ms']
    return dict(calls=len(records), tokens=sum(row['tokens'] for row in records),
                gpu_ms=total, wall_ms=sum(row['wall_ms'] for row in records),
                phases=[dict(name=name, ms=ms, percent=100 * ms / total if total else 0)
                        for name, ms in sorted(phases.items(), key=lambda item: -item[1])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--tokens', type=int, help='Select calls with this exact logged token count')
    parser.add_argument('--skip', type=int, default=0, help='Skip initial parsed calls, e.g. warmup')
    parser.add_argument('--output', type=Path, help='Write selected records and summary as JSON')
    args = parser.parse_args()
    if args.skip < 0 or (args.tokens is not None and args.tokens < 0):
        parser.error('--skip and --tokens must be nonnegative')
    try:
        records = parse_lines(args.log.read_text(errors='replace').splitlines())[args.skip:]
        if args.tokens is not None:
            records = [row for row in records if row['tokens'] == args.tokens]
        if not records:
            parser.error('No matching prefill timing records; enable STRATA_PREFILL_TIMING=1')
        result = dict(note=__doc__.strip(), summary=summarize(records), records=records)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + '\n')
        summary = result['summary']
        print(f"{summary['calls']} calls, {summary['tokens']} logged prefill tokens, "
              f"GPU timeline {summary['gpu_ms']:.0f} ms, wall {summary['wall_ms']:.0f} ms")
        for phase in summary['phases']:
            print(f"{phase['name']:<20} {phase['ms']:9.0f} ms {phase['percent']:6.1f}%")
        print('Diagnostic stream intervals include waits; use uninstrumented runs for throughput.')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()

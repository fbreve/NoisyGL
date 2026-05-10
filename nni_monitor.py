# -*- coding: utf-8 -*-
"""
nni_monitor.py  --  Monitor ao vivo de experimentos NNI.

Uso:
    python nni_monitor.py               # atualiza a cada 20s
    python nni_monitor.py --verbose     # detalhes por trial
    python nni_monitor.py --once        # roda uma vez e sai
    python nni_monitor.py --hours 3     # janela de 3h (default: 2)
    python nni_monitor.py --interval 30 # intervalo em segundos
"""
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse, csv, json, os, re, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NNI_BASE       = Path(os.environ.get('NNI_EXPERIMENTS_DIR', Path.home() / 'nni-experiments'))
NNI_TIMING_DIR = Path('./log')

# Regex
PARAMS_RE      = re.compile(r'Flat params:\s*(\{.+\})')
ACC_RE         = re.compile(r"FINAL_RESULT:\s*(\{.+\})")
PHASE1_DONE_RE = re.compile(r'\[Phase 1\] Done in ([\d.]+)s')
DATASET_RE     = re.compile(r'Name:\s+(\S+)')
NOISE_RE       = re.compile(r'#Actual noise rate ([\d.]+)')
NOISE_TYPE_RE  = re.compile(r'(Pair noise|Random noise|Uniform noise|Instance noise|Clean)')
TIMING_LINE_RE = re.compile(
    r'Trial:\s*(\w+)\s*\|'
    r'\s*Wall:\s*([\d.]+)s\s*\|'
    r'(?:\s*LN-PCC\(CPU\):\s*([\d.]+)s\s*\|)?'
    r'(?:\s*GCN\(GPU\):\s*([\d.]+)s\s*\|)?'
    r'\s*Acc:\s*([\d.]+)\s*\|'
    r'\s*Params:\s*(\{.*\})'
)
NOISE_MAP = {'Pair noise':'pair','Random noise':'random',
             'Uniform noise':'uniform','Instance noise':'instance','Clean':'clean'}


# ── Parse trial.log ───────────────────────────────────────────────────────────
def parse_trial_log(log_path: Path) -> dict:
    info = dict(trial_id=log_path.parent.name, dataset=None,
                noise_type=None, noise_rate=None, params=None,
                acc=None, wall_s=None, pcc_s=None, gcn_s=None,
                status='RUNNING', pcc_done_s=None)
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return info

    m = DATASET_RE.search(text);     info['dataset']    = m.group(1) if m else None
    m = NOISE_TYPE_RE.search(text);  info['noise_type'] = NOISE_MAP.get(m.group(1)) if m else None
    m = NOISE_RE.search(text);       info['noise_rate'] = float(m.group(1)) if m else None
    m = PHASE1_DONE_RE.search(text); info['pcc_done_s'] = float(m.group(1)) if m else None

    m = PARAMS_RE.search(text)
    if m:
        try:    info['params'] = json.loads(m.group(1).replace("'", '"'))
        except: info['params'] = m.group(1)

    m = ACC_RE.search(text)
    if m:
        try:
            res = json.loads(m.group(1).replace("'", '"'))
            info['acc']    = float(res.get('test', 0))
            info['status'] = 'DONE'
        except: pass

    m = TIMING_LINE_RE.search(text)
    if m:
        info['wall_s'] = float(m.group(2))
        info['pcc_s']  = float(m.group(3)) if m.group(3) else None
        info['gcn_s']  = float(m.group(4)) if m.group(4) else None

    return info


# ── Carregar experimentos ─────────────────────────────────────────────────────
def load_experiments(hours_back: float) -> dict:
    """
    Retorna {exp_id -> {'trials': [...], 'active_trial': {...}}}
    O trial ativo é o cujo trial.log tem o mtime mais recente.

    BUG FIX (Windows): O Windows não atualiza o mtime do diretório pai
    quando arquivos filhos são modificados. Por isso, filtramos pelo mtime
    do trial.log mais recente dentro de cada experimento, não pelo mtime
    do diretório do experimento.
    """
    cutoff = datetime.now() - timedelta(hours=hours_back)
    exps = {}
    if not NNI_BASE.exists():
        return exps

    for exp_dir in NNI_BASE.iterdir():
        if not exp_dir.is_dir():
            continue

        trial_root = exp_dir / 'environments' / 'local-env' / 'trials'
        if not trial_root.exists():
            continue

        entries = []  # (mtime_float, Path)
        for td in trial_root.iterdir():
            log = td / 'trial.log'
            if log.exists():
                try:    mtime = log.stat().st_mtime
                except: mtime = 0.0
                entries.append((mtime, log))

        if not entries:
            continue

        entries.sort(key=lambda x: x[0], reverse=True)

        # ── Filtro por mtime do trial.log mais recente (fix Windows) ──────
        latest_log_mtime = datetime.fromtimestamp(entries[0][0])
        if latest_log_mtime < cutoff:
            continue

        active = parse_trial_log(entries[0][1])
        active['log_mtime'] = latest_log_mtime

        all_trials = [active]
        for mtime, log in entries[1:]:
            t = parse_trial_log(log)
            t['log_mtime'] = datetime.fromtimestamp(mtime)
            all_trials.append(t)

        exps[exp_dir.name] = {'trials': all_trials, 'active': active}

    return exps


# ── Carregar nni_timing logs ──────────────────────────────────────────────────
def load_timing_logs() -> list:
    records = []
    for f in NNI_TIMING_DIR.glob('nni_timing_*.log'):
        parts = f.stem.split('_', 2)
        source = parts[2] if len(parts) > 2 else 'unknown'
        try:    text = f.read_text(encoding='utf-8', errors='replace')
        except: continue
        for line in text.splitlines():
            m = TIMING_LINE_RE.match(line.strip())
            if m:
                records.append(dict(
                    source=source, trial_id=m.group(1),
                    wall_s=float(m.group(2)),
                    pcc_s=float(m.group(3)) if m.group(3) else None,
                    gcn_s=float(m.group(4)) if m.group(4) else None,
                    acc=float(m.group(5)), params=m.group(6)))
    return records


# ── Helpers ───────────────────────────────────────────────────────────────────
def progress_bar(done, total, w=22):
    if total == 0: return '[' + '?'*w + ']   ?%'
    f = min(done/total, 1.0)
    return f'[{"#"*int(f*w)+"-"*(w-int(f*w))}] {f*100:5.1f}%'

def fmt_t(s):
    if s is None: return '  N/A '
    return f'{s:5.1f}s' if s < 60 else f'{s/60:5.1f}m'

def fmt_p(p, n=60):
    if p is None: return ''
    s = ' '.join(f'{k}={v:.3f}' if isinstance(v,float) else f'{k}={v}'
                 for k,v in p.items()) if isinstance(p,dict) else str(p)
    return s[:n] + ('...' if len(s)>n else '')


# ── Render ────────────────────────────────────────────────────────────────────
def render(exps: dict, timing: list, now: datetime, args, trials_target: int = 50) -> str:
    lines = []

    # Agrupar por cenario
    scenarios = {}
    for exp_id, exp in exps.items():
        at = exp['active']
        trials = exp['trials']
        ds = at['dataset'] or next((t['dataset'] for t in trials if t['dataset']), None)
        if not ds: continue
        nt = at['noise_type'] or next((t['noise_type'] for t in trials if t['noise_type']), '?')
        nr = at['noise_rate'] if at['noise_rate'] is not None else \
             next((t['noise_rate'] for t in trials if t['noise_rate'] is not None), None)
        key = f'{ds}_{nt}_{nr:.2f}' if nr is not None else f'{ds}_{nt}_?'

        done   = [t for t in trials if t['status']=='DONE']
        best   = max((t['acc'] for t in done if t['acc'] is not None), default=None)
        top3   = sorted(done, key=lambda t: t['acc'] or 0, reverse=True)[:3]

        if key not in scenarios or len(trials) > len(scenarios[key]['trials']):
            scenarios[key] = dict(ds=ds, nt=nt, nr=nr, trials=trials,
                                  done=len(done), best=best, top3=top3, active=at)

    ss = sorted(scenarios.items())
    n_done  = sum(s['done'] for _,s in ss)
    n_all   = sum(len(s['trials']) for _,s in ss)
    n_comp  = sum(1 for _,s in ss if s['done'] >= trials_target)
    n_pcc   = sum(1 for _,s in ss if s['active']['dataset'] and not s['active']['pcc_done_s'])
    n_gcn   = sum(1 for _,s in ss if s['active']['dataset'] and s['active']['pcc_done_s'])

    SEP = '=' * 98
    lines += [
        SEP,
        f"  NNI Monitor  [{now.strftime('%Y-%m-%d %H:%M:%S')}]  |  "
        f"{len(ss)} cenarios  |  {n_comp} completos  |  "
        f"{n_done}/{n_all} trials done  |  "
        f"ATIVO: {n_pcc} PCC(CPU)  {n_gcn} GCN(GPU)",
        SEP,
        f"  {'Cenario':<36}  {'Progresso':<29}  "
        f"{'OK':>4}  {'Tot':>4}  {'Melhor':>8}  Trial ativo",
        '-' * 98,
    ]

    for key, s in ss:
        bar  = progress_bar(s['done'], max(len(s['trials']), trials_target))
        best = f"{s['best']:.4f}" if s['best'] is not None else '   N/A '
        nr   = s['nr']
        lbl  = f"{s['ds']} {s['nt']} {nr:.2f}" if nr is not None else key

        at = s['active']
        if at['params'] is not None:
            mt = at.get('log_mtime', now).strftime('%H:%M')
            if at['pcc_done_s']:
                st = f"[GCN] pcc={fmt_t(at['pcc_done_s'])} @{mt}"
            else:
                st = f"[PCC] {fmt_p(at['params'], 38)} @{mt}"
        elif at['status'] == 'DONE':
            st = f"DONE acc={at['acc']:.4f}" if at['acc'] else 'DONE'
        else:
            st = '(aguardando...)'

        lines.append(f"  {lbl:<36}  {bar}  {s['done']:>4}  {len(s['trials']):>4}  {best:>8}  {st}")

        if args.verbose:
            at = s['active']
            if at['params']:
                ph = 'GCN' if at['pcc_done_s'] else 'PCC'
                pi = f"pcc={fmt_t(at['pcc_done_s'])}" if at['pcc_done_s'] else 'pcc=em andamento'
                lines.append(f"    [ATIVO {ph}] {at['trial_id']}  {pi}")
                lines.append(f"      {fmt_p(at['params'], 80)}")
            for i, t in enumerate(s['top3'], 1):
                a = f"{t['acc']:.4f}" if t['acc'] else 'N/A'
                lines.append(
                    f"    #{i} DONE {t['trial_id']}  acc={a}  "
                    f"wall={fmt_t(t['wall_s'])}  pcc={fmt_t(t['pcc_done_s'] or t['pcc_s'])}  "
                    f"gcn={fmt_t(t['gcn_s'])}"
                )
                lines.append(f"      {fmt_p(t['params'], 80)}")

    # nni_timing logs
    if timing:
        lines += ['', '-- nni_timing logs ' + '-'*60]
        by_src = defaultdict(list)
        for r in timing: by_src[r['source']].append(r)
        for src, recs in sorted(by_src.items()):
            accs  = [r['acc'] for r in recs if r['acc'] is not None]
            best  = f'{max(accs):.4f}' if accs else 'N/A'
            walls = [r['wall_s'] for r in recs if r['wall_s']]
            avgw  = f"{sum(walls)/len(walls):.1f}s" if walls else 'N/A'
            lines.append(f"  {src:<40}  {len(recs):>3} trials  best={best}  avg_wall={avgw}")
            for r in recs[-2:]:
                lines.append(
                    f"    {r['trial_id']}  acc={r['acc']:.4f}  "
                    f"wall={fmt_t(r['wall_s'])}  pcc={fmt_t(r['pcc_s'])}  gcn={fmt_t(r['gcn_s'])}"
                )
                lines.append(f"      {str(r['params'])[:80]}")

    lines.append('')
    lines.append(f"  [Atualiza a cada {args.interval}s -- Ctrl+C para parar]")
    return '\n'.join(lines)


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(exps: dict, csv_path: Path):
    rows = []
    for exp_id, exp in exps.items():
        for t in exp['trials']:
            if t['status'] != 'DONE': continue
            rows.append(dict(
                exp_id=exp_id, trial_id=t['trial_id'],
                dataset=t['dataset'], noise_type=t['noise_type'], noise_rate=t['noise_rate'],
                acc=t['acc'], wall_s=t['wall_s'],
                pcc_s=t['pcc_done_s'] or t['pcc_s'], gcn_s=t['gcn_s'],
                params=json.dumps(t['params']) if isinstance(t['params'],dict) else t['params'],
            ))
    if not rows: return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hours',           type=float, default=2.0,
                   help='Janela de tempo em horas para filtrar experimentos (default: 2)')
    p.add_argument('--interval',        type=float, default=20.0)
    p.add_argument('--no_clear',        action='store_true')
    p.add_argument('--verbose',  '-v',  action='store_true')
    p.add_argument('--once',            action='store_true')
    p.add_argument('--output',          type=str, default=None)
    p.add_argument('--optimize_trials', type=int, default=50,
                   help='Numero de trials por cenario (para calcular completude, default: 50)')
    args = p.parse_args()

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = Path(args.output) if args.output else Path(f'log/nni_monitor_{ts}.csv')

    print(f'[monitor] NNI base : {NNI_BASE}')
    print(f'[monitor] Janela   : {args.hours}h  |  intervalo: {args.interval}s')
    print(f'[monitor] CSV      : {csv_path}')
    if not NNI_BASE.exists():
        print(f'[monitor] ERRO: {NNI_BASE} nao encontrado'); sys.exit(1)

    it = 0
    try:
        while True:
            now    = datetime.now()
            exps   = load_experiments(args.hours)
            timing = load_timing_logs()
            out    = render(exps, timing, now, args, trials_target=args.optimize_trials)
            if not args.no_clear and it > 0:
                print('\033[2J\033[H', end='', flush=True)
            print(out, flush=True)
            save_csv(exps, csv_path)
            if args.once: break
            it += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n[monitor] Parado. CSV:', csv_path)

if __name__ == '__main__':
    main()

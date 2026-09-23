import json, re
from pathlib import Path
BASE = Path("/data/rpent/logs/eval_object_swap")
print(f"{'ep':8} {'res':4} {'el_s':6} {'req':4} {'ctx_max':7} {'ctx_avg':7} {'prefill_new':10} {'cache_hit%':9} {'decode':7} {'turn_s':6}")
agg = {"el": [], "req": [], "ctx": [], "pf": [], "ch": [], "dc": [], "tn": []}
for d in sorted(BASE.glob("t*_s*")):
    log = d / "run.log"
    if not log.exists(): continue
    text = log.read_text(errors="ignore")
    us = re.findall(r"(\d\d:\d\d:\d\d) I \[api_loop\] \[usage\] in=(\d+) out=(\d+) cache_read=(\d+)", text)
    if not us: continue
    from datetime import datetime
    ts = [datetime.strptime(u[0], "%H:%M:%S") for u in us]
    ins = [int(u[1]) for u in us]; outs = [int(u[2]) for u in us]; cas = [int(u[3]) for u in us]
    # diffs = per-request prompt context / decode tokens
    ctx_per = [ins[0]] + [ins[i]-ins[i-1] for i in range(1, len(ins))]
    out_per = [outs[0]] + [outs[i]-outs[i-1] for i in range(1, len(outs))]
    cache_per = [cas[0]] + [cas[i]-cas[i-1] for i in range(1, len(cas))]
    prefill_new_per = [c - k for c, k in zip(ctx_per, cache_per)]
    cache_hit = 100 * sum(cache_per) / max(sum(ctx_per), 1)
    dts = [(ts[i]-ts[i-1]).total_seconds() for i in range(1, len(ts))]
    turn_avg = sum(dts)/len(dts) if dts else 0
    solved = "recipe: object_swap" in text and "not written" not in text.split("recipe:")[-1]
    solved = bool(re.search(r"I \[agent\] recipe: \S+_recipe", text))
    el = re.findall(r"elapsed: ([\d.]+)s", text)
    print(f"{d.name:8} {'OK' if solved else 'FAIL':4} {el[-1] if el else '-':6} {len(us):4} {max(ctx_per):7} {sum(ctx_per)//len(ctx_per):7} {sum(prefill_new_per):10} {cache_hit:9.1f} {sum(out_per):7} {turn_avg:6.1f}")
    agg["el"].append(float(el[-1]) if el else 0); agg["req"].append(len(us))
    agg["ctx"].append(max(ctx_per)); agg["pf"].append(sum(prefill_new_per))
    agg["ch"].append(cache_hit); agg["dc"].append(sum(out_per)); agg["tn"].append(turn_avg)
n = len(agg["el"]); ok = [i for i, d in enumerate(sorted(BASE.glob('t*_s*')))]
print(f"\n== 均值 over {n} 集 ==")
print(f"单集时长 {sum(agg['el'])/n:.0f}s | 请求/集 {sum(agg['req'])/n:.0f} | 单请求上下文 max {max(agg['ctx'])//1000}k avg {sum(agg['ctx'])//n//1000}k")
print(f"每集新增 prefill {sum(agg['pf'])/n/1000:.0f}k tok | cache 命中率 {sum(agg['ch'])/n:.1f}% | 每集 decode {sum(agg['dc'])/n/1000:.1f}k tok")
print(f"平均 turn 周期（含工具执行）{sum(agg['tn'])/n:.1f}s")

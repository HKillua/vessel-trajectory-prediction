"""MambaDiff-ECR 训练分析报告生成器

读取训练输出目录中的 log.txt / phase1_metrics.csv / phase2_metrics.csv / config_snapshot.yaml,
生成一份结构化的 Markdown 诊断报告。

Usage:
    python -m mambadiff.analyze --log_dir results/mambadiff
"""

import argparse
import csv
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Utility: ASCII sparkline chart
# ---------------------------------------------------------------------------

def ascii_chart(values, width=60, height=12, label="", fmt=".4f"):
    if not values or all(v is None for v in values):
        return f"  (no data for {label})\n"

    clean = [(i, v) for i, v in enumerate(values) if v is not None and not math.isnan(v)]
    if not clean:
        return f"  (all NaN for {label})\n"

    indices, vals = zip(*clean)
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-8

    n = len(values)
    bucket_size = max(1, n / width)
    cols = []
    for c in range(width):
        lo = int(c * bucket_size)
        hi = int((c + 1) * bucket_size)
        hi = max(hi, lo + 1)
        bucket_vals = [v for i, v in clean if lo <= i < hi]
        if bucket_vals:
            cols.append(statistics.mean(bucket_vals))
        else:
            cols.append(None)

    lines = []
    for row in range(height - 1, -1, -1):
        threshold = vmin + (vmax - vmin) * row / (height - 1)
        line = ""
        for c in cols:
            if c is None:
                line += " "
            elif c >= threshold:
                line += "█"
            else:
                line += " "
        y_label = f"{threshold:{fmt}}" if row in (0, height - 1, height // 2) else ""
        lines.append(f"  {y_label:>10s} │{line}│")

    x_axis = f"  {'':>10s} └{'─' * width}┘"
    x_labels = f"  {'':>10s}  {'0':>{1}}{'':>{width - 6}}{n - 1}"
    title = f"  {label}" if label else ""

    return "\n".join([title] + lines + [x_axis, x_labels]) + "\n"


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------

def parse_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                k = k.strip()
                if v is None or v.strip() == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v.strip())
                    except ValueError:
                        parsed[k] = v.strip()
            rows.append(parsed)
    return rows


# ---------------------------------------------------------------------------
# Log Parsing
# ---------------------------------------------------------------------------

def parse_log(path):
    if not os.path.exists(path):
        return {}

    result = {
        'anomalies': [],
        'cfg_health': [],
        'grad_norms': [],
        'sda_gates': [],
        'early_stop': None,
        'div_protection': [],
        'val_metrics': [],
        'stratified': [],
        'phase_markers': [],
        'gpu_peak': None,
        'device': None,
        'train_samples': None,
        'val_samples': None,
        'test_samples': None,
        'model_params': {},
        'encounter_dist': {},
        'ships_per_scene': None,
    }

    ts_re = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)$')
    anomaly_re = re.compile(r'\*\*\* ANOMALY: (.+?) \*\*\*')
    cfg_re = re.compile(
        r'CFG health: ADE_base=([\d.]+) ADE_cfg=([\d.]+) delta=([+-][\d.]+) (\[.+?\])')
    grad_re = re.compile(r'grad_norms \[(\d+)\]: (.+)')
    sda_re = re.compile(r'sda_gates \[(\d+)\]: (.+)')
    early_re = re.compile(r'Early stopping: no improvement for (\d+) evals')
    no_improve_re = re.compile(
        r'No improvement (\d+)/(\d+) \(combined=([\d.]+), best=([\d.]+)\)')
    div_protect_re = re.compile(r'reducing div_weight to ([\d.]+)')
    val_re = re.compile(r'Val: ADE=([\d.]+)\s+FDE=([\d.]+)')
    diff_re = re.compile(r'Diffusion_improvement=([-\d.]+)%')
    ccr_re = re.compile(r'CCR=([\d.]+)%\((\d+)\)')
    device_re = re.compile(r'^Device: (.+)$')
    samples_re = re.compile(r'^(Train|Val|Test) samples: (\d+)$')
    param_re = re.compile(r'^\s{2}(\S+)\s*:\s*([\d,]+)\s*/\s*([\d,]+)$')
    total_param_re = re.compile(r'Trainable/Total: ([\d,]+)/([\d,]+)')
    enc_dist_re = re.compile(r'^(\w+) encounter distribution \((\d+) samples\): (.+)$')
    ships_re = re.compile(r'ships/scene: mean=([\d.]+) median=([\d.]+) max=(\d+)')
    gpu_peak_re = re.compile(r'Peak GPU memory after 1st batch: (\d+) MB')
    phase_re = re.compile(r'^=== (Phase \d+: .+?) ===$')
    strat_re = re.compile(r'^\s+([\w_]+): ADE=([\d.]+) FDE=([\d.]+) \(n=(\d+)\)')
    best_re = re.compile(r'Best ADE=([\d.]+)\s+Best CCR=([\d.]+)%')

    with open(path) as f:
        for line in f:
            m = ts_re.match(line.strip())
            if not m:
                continue
            ts, msg = m.group(1), m.group(2)

            am = anomaly_re.search(msg)
            if am:
                result['anomalies'].append({'ts': ts, 'msg': am.group(1)})

            cm = cfg_re.search(msg)
            if cm:
                result['cfg_health'].append({
                    'base': float(cm.group(1)),
                    'cfg': float(cm.group(2)),
                    'delta': float(cm.group(3)),
                    'status': cm.group(4),
                })

            gm = grad_re.search(msg)
            if gm:
                step = int(gm.group(1))
                parts = re.findall(r'(\w+)=([\d.]+)', gm.group(2))
                result['grad_norms'].append({
                    'step': step,
                    'norms': {k: float(v) for k, v in parts},
                })

            sm = sda_re.search(msg)
            if sm:
                step = int(sm.group(1))
                layers = re.findall(r'L(\d+)=\[([\d.,]+)\]', sm.group(2))
                gate_dict = {}
                for li, vals in layers:
                    gate_dict[f'L{li}'] = [float(x) for x in vals.split(',')]
                result['sda_gates'].append({'step': step, 'gates': gate_dict})

            em = early_re.search(msg)
            if em:
                result['early_stop'] = int(em.group(1))

            nim = no_improve_re.search(msg)
            if nim:
                result['no_improve_last'] = {
                    'count': int(nim.group(1)),
                    'patience': int(nim.group(2)),
                    'combined': float(nim.group(3)),
                    'best': float(nim.group(4)),
                }

            dm = div_protect_re.search(msg)
            if dm:
                result['div_protection'].append(float(dm.group(1)))

            vm = val_re.search(msg)
            if vm:
                entry = {'ade': float(vm.group(1)), 'fde': float(vm.group(2))}
                di = diff_re.search(msg)
                if di:
                    entry['diff_improve'] = float(di.group(1))
                cr = ccr_re.search(msg)
                if cr:
                    entry['ccr'] = float(cr.group(1))
                    entry['ccr_n'] = int(cr.group(2))
                result['val_metrics'].append(entry)

            dvm = device_re.match(msg)
            if dvm:
                result['device'] = dvm.group(1)

            spm = samples_re.match(msg)
            if spm:
                result[f'{spm.group(1).lower()}_samples'] = int(spm.group(2))

            tpm = total_param_re.search(msg)
            if tpm:
                result['total_params'] = int(tpm.group(1).replace(',', ''))

            pm = param_re.match(msg)
            if pm:
                result['model_params'][pm.group(1)] = int(pm.group(2).replace(',', ''))

            edm = enc_dist_re.match(msg)
            if edm:
                split_name = edm.group(1)
                pairs = re.findall(r'(\w[\w-]*)=(\d+)', edm.group(3))
                result['encounter_dist'][split_name] = {k: int(v) for k, v in pairs}

            shm = ships_re.search(msg)
            if shm:
                result['ships_per_scene'] = {
                    'mean': float(shm.group(1)),
                    'median': float(shm.group(2)),
                    'max': int(shm.group(3)),
                }

            gpm = gpu_peak_re.search(msg)
            if gpm:
                result['gpu_peak'] = int(gpm.group(1))

            phm = phase_re.match(msg)
            if phm:
                result['phase_markers'].append({'ts': ts, 'phase': phm.group(1)})

            stm = strat_re.match(msg)
            if stm:
                result['stratified'].append({
                    'category': stm.group(1),
                    'ade': float(stm.group(2)),
                    'fde': float(stm.group(3)),
                    'n': int(stm.group(4)),
                })

            bm = best_re.search(msg)
            if bm:
                result['best_final'] = {
                    'ade': float(bm.group(1)),
                    'ccr': float(bm.group(2)),
                }

    return result


# ---------------------------------------------------------------------------
# Config Parsing (no yaml dependency)
# ---------------------------------------------------------------------------

def parse_config(path):
    if not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    # Fallback: simple key-value extraction
    cfg = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^\s{2}(\w+):\s*(.+)$', line)
            if m:
                k, v = m.group(1), m.group(2).strip()
                try:
                    v = float(v)
                    if v == int(v):
                        v = int(v)
                except ValueError:
                    pass
                cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Diagnostic Rules
# ---------------------------------------------------------------------------

def run_diagnostics(p1, p2, log_info, cfg):
    issues = []
    suggestions = []

    # 1. Phase 1 convergence
    if p1:
        losses = [r.get('train_noise_loss') for r in p1
                  if r.get('train_noise_loss') is not None]
        if len(losses) >= 10:
            first_10pct = statistics.mean(losses[:max(1, len(losses) // 10)])
            last_10pct = statistics.mean(losses[-max(1, len(losses) // 10):])
            if last_10pct > first_10pct * 0.9:
                issues.append(
                    "⚠️ Phase 1 noise loss未充分收敛 "
                    f"(前10%均值={first_10pct:.4f}, 后10%均值={last_10pct:.4f})")
                suggestions.append("考虑增加Phase 1 epochs或调整学习率")

    # 2. Phase 2 loss dominance
    if p2:
        last_rows = p2[-max(1, len(p2) // 5):]
        components = ['dist_loss', 'fredf_loss', 'anchor_loss', 'nomoto_loss', 'div_loss']
        totals = defaultdict(float)
        count = 0
        for r in last_rows:
            row_total = 0
            for c in components:
                v = r.get(c)
                if v is not None and not math.isnan(v):
                    totals[c] += abs(v)
                    row_total += abs(v)
            if row_total > 0:
                count += 1
        if count > 0:
            grand_total = sum(totals.values())
            if grand_total > 0:
                for c in components:
                    ratio = totals[c] / grand_total
                    if ratio > 0.60:
                        name = c.replace('_loss', '')
                        issues.append(
                            f"⚠️ {name} loss占比过高 ({ratio:.0%})，主导了训练梯度")
                        suggestions.append(
                            f"降低 loss_{name}_weight 或提高其他loss权重以平衡梯度")

    # 3. ADE improvement
    if p2:
        val_ades = [(i, r['val_ade']) for i, r in enumerate(p2)
                    if r.get('val_ade') is not None and not math.isnan(r['val_ade'])]
        if len(val_ades) >= 6:
            first_vals = [v for _, v in val_ades[:3]]
            last_vals = [v for _, v in val_ades[-3:]]
            if statistics.mean(last_vals) >= statistics.mean(first_vals) * 0.95:
                issues.append(
                    "⚠️ ADE几乎没有改善 "
                    f"(早期={statistics.mean(first_vals):.4f}, "
                    f"晚期={statistics.mean(last_vals):.4f})")
                suggestions.append(
                    "检查Phase 1是否充分收敛；尝试降低diversity_weight；检查数据质量")

    # 4. Diffusion contribution
    if p2:
        pairs = [(r.get('val_ade'), r.get('val_anchor_ade')) for r in p2
                 if r.get('val_ade') is not None and r.get('val_anchor_ade') is not None
                 and not math.isnan(r['val_ade']) and not math.isnan(r['val_anchor_ade'])]
        if pairs:
            last_pair = pairs[-1]
            ade, anchor_ade = last_pair
            if anchor_ade > 0:
                improvement = (anchor_ade - ade) / anchor_ade * 100
                if improvement < 5:
                    issues.append(
                        f"⚠️ 扩散去噪几乎没有改善轨迹 "
                        f"(Anchor ADE={anchor_ade:.4f}, Final ADE={ade:.4f}, "
                        f"改善仅{improvement:.1f}%)")
                    suggestions.append(
                        "检查denoiser是否在Phase 1充分训练；"
                        "检查tau_start设置是否合理；"
                        "尝试增加num_tau步数")

    # 5. CCR
    val_metrics = log_info.get('val_metrics', [])
    if val_metrics:
        last_ccr = [m.get('ccr') for m in val_metrics[-3:] if m.get('ccr') is not None]
        if last_ccr and statistics.mean(last_ccr) < 70:
            issues.append(
                f"⚠️ COLREGs合规率偏低 (近期CCR={statistics.mean(last_ccr):.1f}%)")
            suggestions.append(
                "增大 guidance.colregs_weight 或 training.colregs_aux_weight")

    # 6. CFG health
    cfg_checks = log_info.get('cfg_health', [])
    if cfg_checks:
        n_hurt = sum(1 for c in cfg_checks if c['delta'] > 0)
        if n_hurt > len(cfg_checks) * 0.5:
            issues.append(
                f"⚠️ CFG多次导致ADE恶化 ({n_hurt}/{len(cfg_checks)}次)")
            suggestions.append(
                "增大 diffusion.cfg_dropout (当前可能学习不到好的无条件分布)；"
                "或在推理时不使用CFG")

    # 7. Anomalies
    anomalies = log_info.get('anomalies', [])
    if anomalies:
        nan_count = sum(1 for a in anomalies if 'NaN' in a['msg'] or 'Inf' in a['msg'])
        spike_count = sum(1 for a in anomalies if 'spiked' in a['msg'])
        if nan_count:
            issues.append(f"🔴 训练中出现 {nan_count} 次NaN/Inf异常")
            suggestions.append("检查学习率是否过大；检查数据是否有异常值；尝试关闭AMP")
        if spike_count:
            issues.append(f"⚠️ 训练中出现 {spike_count} 次loss突刺(>5x)")
            suggestions.append("可能是某个异常batch导致的，检查数据预处理")

    # 8. Diversity protection
    div_protections = log_info.get('div_protection', [])
    if div_protections:
        issues.append(
            f"⚠️ Diversity权重被自动降低了 {len(div_protections)} 次 "
            f"(当前={div_protections[-1]:.4f})")
        suggestions.append("Diversity loss和accuracy存在冲突，考虑降低初始diversity_weight")

    # 9. Early stopping
    early = log_info.get('early_stop')
    if early is not None and p2:
        total_epochs = len(p2)
        configured_epochs = cfg.get('training', {}).get('num_epochs', 100) if isinstance(cfg, dict) else 100
        if total_epochs < configured_epochs * 0.3:
            issues.append(
                f"⚠️ Early stopping在第{total_epochs}轮就触发了"
                f"(计划{configured_epochs}轮，仅完成{total_epochs / configured_epochs:.0%})")
            suggestions.append("模型可能过拟合或学习率衰减过快；增大patience或调整lr schedule")

    # 10. No val/test data
    if log_info.get('val_samples') == 0 or log_info.get('val_samples') is None:
        issues.append("🔴 没有验证集！训练期间无法评估模型性能")
        suggestions.append("确保数据预处理生成了val/test分割")

    return issues, suggestions


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(log_dir):
    p1 = parse_csv(os.path.join(log_dir, 'phase1_metrics.csv'))
    p2 = parse_csv(os.path.join(log_dir, 'phase2_metrics.csv'))
    log_info = parse_log(os.path.join(log_dir, 'log.txt'))
    cfg = parse_config(os.path.join(log_dir, 'config_snapshot.yaml'))

    lines = []
    w = lines.append

    w("# MambaDiff-ECR 训练分析报告")
    w(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"日志目录: `{os.path.abspath(log_dir)}`")

    # ---- 1. 训练概览 ----
    w("\n## 1. 训练概览\n")
    w(f"| 项目 | 值 |")
    w(f"|------|------|")
    w(f"| 设备 | {log_info.get('device', 'N/A')} |")
    w(f"| 训练样本数 | {log_info.get('train_samples', 'N/A')} |")
    w(f"| 验证样本数 | {log_info.get('val_samples', 'N/A')} |")
    w(f"| 测试样本数 | {log_info.get('test_samples', 'N/A')} |")
    w(f"| 模型参数量 | {log_info.get('total_params', 'N/A'):,} |"
      if log_info.get('total_params') else "| 模型参数量 | N/A |")
    w(f"| GPU峰值显存 | {log_info.get('gpu_peak', 'N/A')} MB |")
    w(f"| Phase 1 epochs | {len(p1)} |")
    w(f"| Phase 2 epochs | {len(p2)} |")

    if p1:
        total_time_p1 = sum(r.get('epoch_time_s', 0) or 0 for r in p1)
        w(f"| Phase 1 总时间 | {total_time_p1:.0f}s ({total_time_p1/60:.1f}min) |")
    if p2:
        total_time_p2 = sum(r.get('epoch_time_s', 0) or 0 for r in p2)
        w(f"| Phase 2 总时间 | {total_time_p2:.0f}s ({total_time_p2/60:.1f}min) |")

    if log_info.get('ships_per_scene'):
        sp = log_info['ships_per_scene']
        w(f"| 场景船数 | mean={sp['mean']:.1f} median={sp['median']} max={sp['max']} |")

    if log_info.get('early_stop') is not None:
        w(f"| Early stopping | 第{len(p2)}轮触发 (patience={log_info.get('no_improve_last', {}).get('patience', '?')}) |")

    # Best metrics
    best_ade = None
    best_fde = None
    best_ccr = None
    if p2:
        val_ades = [(i, r['val_ade']) for i, r in enumerate(p2)
                    if r.get('val_ade') is not None and not math.isnan(r['val_ade'])]
        if val_ades:
            best_i, best_ade = min(val_ades, key=lambda x: x[1])
            w(f"| **Best ADE** | **{best_ade:.4f} nm** (epoch {int(p2[best_i].get('epoch', best_i))}) |")
        val_fdes = [(i, r['val_fde']) for i, r in enumerate(p2)
                    if r.get('val_fde') is not None and not math.isnan(r['val_fde'])]
        if val_fdes:
            best_i, best_fde = min(val_fdes, key=lambda x: x[1])
            w(f"| **Best FDE** | **{best_fde:.4f} nm** (epoch {int(p2[best_i].get('epoch', best_i))}) |")
        val_ccrs = [(i, r['val_ccr']) for i, r in enumerate(p2)
                    if r.get('val_ccr') is not None and not math.isnan(r['val_ccr'])]
        if val_ccrs:
            best_i, best_ccr = max(val_ccrs, key=lambda x: x[1])
            w(f"| **Best CCR** | **{best_ccr:.4f}** (epoch {int(p2[best_i].get('epoch', best_i))}) |")

    # ---- 2. 配置摘要 ----
    w("\n## 2. 关键配置\n")
    if isinstance(cfg, dict) and cfg:
        tc = cfg.get('training', {})
        mc = cfg.get('model', {})
        dc = cfg.get('diffusion', {})
        gc = cfg.get('guidance', {})
        w("```")
        w(f"encoder_type: {mc.get('encoder_type', '?')}  |  d_model: {mc.get('d_model', '?')}  |  k_samples: {mc.get('k_samples', '?')}")
        w(f"obs/pred steps: {mc.get('obs_steps', '?')}/{mc.get('pred_steps', '?')}  |  pred_dim: {mc.get('pred_dim', '?')}")
        w(f"diffusion steps: {dc.get('steps', '?')}  |  tau_start: {dc.get('tau_start', '?')}  |  num_tau: {dc.get('num_tau', '?')}  |  cfg_dropout: {dc.get('cfg_dropout', '?')}")
        w(f"lr: {tc.get('lr', '?')}  |  batch_size: {tc.get('batch_size', '?')}  |  num_epochs: {tc.get('num_epochs', '?')}")
        w(f"loss weights: dist={tc.get('loss_dist_weight', '?')} fredf={tc.get('loss_fredf_weight', '?')} anchor={tc.get('loss_anchor_weight', '?')} div={tc.get('loss_diversity_weight', '?')}")
        w(f"colregs_weight: {gc.get('colregs_weight', '?')}  |  colregs_aux_weight: {tc.get('colregs_aux_weight', '?')}")
        fc = cfg.get('fredf', {})
        w(f"fredf: log_mag={fc.get('log_magnitude', '?')} low_w={fc.get('low_weight', '?')} high_w={fc.get('high_weight', '?')}")
        w("```")
    else:
        w("(配置文件未找到或解析失败)")

    # ---- 3. 数据分布 ----
    enc_dist = log_info.get('encounter_dist', {})
    if enc_dist:
        w("\n## 3. 数据分布\n")
        for split_name, counts in enc_dist.items():
            total = sum(counts.values())
            parts = [f"{k}={v}({v/total*100:.0f}%)" for k, v in sorted(counts.items())]
            w(f"**{split_name}** ({total} pairs): {', '.join(parts)}")

    # ---- 4. Phase 1 分析 ----
    w("\n## 4. Phase 1: Denoiser预训练\n")
    if p1:
        train_losses = [r.get('train_noise_loss') for r in p1]
        val_losses = [r.get('val_noise_loss') for r in p1]

        w("### Train Noise Loss")
        w("```")
        w(ascii_chart(train_losses, label="train_noise_loss", fmt=".4f"))
        w("```")

        val_clean = [v for v in val_losses if v is not None and not math.isnan(v)]
        if val_clean:
            w("### Val Noise Loss")
            w("```")
            w(ascii_chart(val_losses, label="val_noise_loss", fmt=".4f"))
            w("```")
            w(f"Best val noise loss: **{min(val_clean):.6f}**")

        if train_losses:
            first = train_losses[0]
            last = train_losses[-1]
            if first and last:
                w(f"\n收敛: {first:.6f} → {last:.6f} (降低{(1-last/first)*100:.1f}%)")
    else:
        w("(无Phase 1数据)")

    # ---- 5. Phase 2 损失分析 ----
    w("\n## 5. Phase 2: 端到端训练\n")
    if p2:
        w("### 总Loss曲线")
        w("```")
        w(ascii_chart([r.get('total_loss') for r in p2], label="total_loss", fmt=".4f"))
        w("```")

        w("### 各损失组件 (最后20%均值)")
        last_n = max(1, len(p2) // 5)
        last_rows = p2[-last_n:]
        comp_names = [
            ('dist_loss', 'dist'),
            ('fredf_loss', 'fredf'),
            ('anchor_loss', 'anchor'),
            ('nomoto_loss', 'nomoto'),
            ('div_loss', 'diversity'),
        ]
        comp_avgs = {}
        for csv_name, display in comp_names:
            vals = [r.get(csv_name) for r in last_rows
                    if r.get(csv_name) is not None and not math.isnan(r[csv_name])]
            if vals:
                comp_avgs[display] = statistics.mean(vals)

        if comp_avgs:
            grand = sum(abs(v) for v in comp_avgs.values())
            w("```")
            for name, avg in comp_avgs.items():
                pct = abs(avg) / grand * 100 if grand > 0 else 0
                bar = "█" * int(pct / 2)
                w(f"  {name:>10s}: {avg:>10.4f}  ({pct:5.1f}%) {bar}")
            w("```")

        # Individual loss curves
        for csv_name, display in comp_names:
            vals = [r.get(csv_name) for r in p2]
            has_data = any(v is not None and not math.isnan(v) for v in vals if v is not None)
            if has_data:
                w(f"\n### {display} loss")
                w("```")
                w(ascii_chart(vals, label=display, fmt=".4f"))
                w("```")
    else:
        w("(无Phase 2数据)")

    # ---- 6. 验证指标 ----
    w("\n## 6. 验证指标趋势\n")
    if p2:
        val_ades = [r.get('val_ade') for r in p2]
        val_fdes = [r.get('val_fde') for r in p2]
        has_val = any(v is not None and not math.isnan(v) for v in val_ades if v is not None)

        if has_val:
            w("### ADE (海里)")
            w("```")
            w(ascii_chart(val_ades, label="val_ADE", fmt=".4f"))
            w("```")

            w("### FDE (海里)")
            w("```")
            w(ascii_chart(val_fdes, label="val_FDE", fmt=".4f"))
            w("```")

            # CCR
            val_ccrs = [r.get('val_ccr') for r in p2]
            has_ccr = any(v is not None and not math.isnan(v) for v in val_ccrs if v is not None)
            if has_ccr:
                w("### CCR (COLREGs合规率)")
                w("```")
                w(ascii_chart(val_ccrs, label="val_CCR", fmt=".4f"))
                w("```")
        else:
            w("(无验证指标 — 缺少验证集)")

    # ---- 7. 扩散贡献分析 ----
    w("\n## 7. 扩散贡献分析\n")
    if p2:
        pairs = [(r.get('val_ade'), r.get('val_anchor_ade')) for r in p2
                 if r.get('val_ade') is not None and r.get('val_anchor_ade') is not None
                 and not math.isnan(r.get('val_ade', float('nan')))
                 and not math.isnan(r.get('val_anchor_ade', float('nan')))]
        if pairs:
            w("| Epoch区间 | Anchor ADE | Final ADE | 改善 |")
            w("|-----------|-----------|-----------|------|")
            chunk_size = max(1, len(pairs) // 5)
            for i in range(0, len(pairs), chunk_size):
                chunk = pairs[i:i+chunk_size]
                avg_ade = statistics.mean(a for a, _ in chunk)
                avg_anchor = statistics.mean(b for _, b in chunk)
                improve = (avg_anchor - avg_ade) / avg_anchor * 100 if avg_anchor > 0 else 0
                w(f"| {i}-{min(i+chunk_size, len(pairs))-1} | {avg_anchor:.4f} | {avg_ade:.4f} | {improve:+.1f}% |")

            last_ade, last_anchor = pairs[-1]
            final_improve = (last_anchor - last_ade) / last_anchor * 100 if last_anchor > 0 else 0
            w(f"\n最终: Anchor={last_anchor:.4f} → Diffusion={last_ade:.4f} (**{final_improve:+.1f}%**)")
        else:
            w("(无anchor对比数据)")
    else:
        w("(无数据)")

    # ---- 8. CFG 健康 ----
    w("\n## 8. CFG健康检查\n")
    cfg_checks = log_info.get('cfg_health', [])
    if cfg_checks:
        n_ok = sum(1 for c in cfg_checks if c['delta'] < 0)
        n_hurt = len(cfg_checks) - n_ok
        w(f"共 {len(cfg_checks)} 次检查: ✅ 有效={n_ok}次, ❌ 有害={n_hurt}次\n")
        w("| 次数 | Base ADE | CFG ADE | Delta | 状态 |")
        w("|------|---------|---------|-------|------|")
        for i, c in enumerate(cfg_checks):
            w(f"| {i+1} | {c['base']:.4f} | {c['cfg']:.4f} | {c['delta']:+.4f} | {c['status']} |")
    else:
        w("(无CFG检查数据 — 训练未达到epoch 10或无验证集)")

    # ---- 9. 异常事件 ----
    w("\n## 9. 异常事件\n")
    anomalies = log_info.get('anomalies', [])
    if anomalies:
        w(f"共 {len(anomalies)} 个异常:\n")
        for a in anomalies:
            w(f"- `[{a['ts']}]` {a['msg']}")
    else:
        w("✅ 无异常事件")

    # ---- 10. SDA门控 ----
    w("\n## 10. SDA门控演化\n")
    sda = log_info.get('sda_gates', [])
    if sda:
        first = sda[0]['gates']
        last = sda[-1]['gates']
        w("| 层 | 初始值 | 最终值 | 变化 |")
        w("|----|--------|--------|------|")
        for layer_key in sorted(first.keys()):
            init_vals = first[layer_key]
            final_vals = last.get(layer_key, init_vals)
            init_str = ",".join(f"{v:.3f}" for v in init_vals)
            final_str = ",".join(f"{v:.3f}" for v in final_vals)
            delta = statistics.mean(abs(f - i) for f, i in zip(final_vals, init_vals))
            w(f"| {layer_key} | [{init_str}] | [{final_str}] | Δ={delta:.4f} |")
    else:
        w("(无SDA数据)")

    # ---- 11. Stratified Results ----
    strat = log_info.get('stratified', [])
    if strat:
        w("\n## 11. 分层评估\n")
        w("| 类别 | ADE | FDE | 样本数 |")
        w("|------|-----|-----|--------|")
        for s in strat:
            w(f"| {s['category']} | {s['ade']:.4f} | {s['fde']:.4f} | {s['n']} |")

    # ---- 12. 诊断建议 ----
    w("\n## 12. 自动诊断\n")
    issues, suggestions = run_diagnostics(p1, p2, log_info, cfg)

    if not issues:
        w("✅ 未发现明显问题")
    else:
        w("### 发现的问题\n")
        for i, issue in enumerate(issues, 1):
            w(f"{i}. {issue}")

        w("\n### 建议\n")
        for i, sug in enumerate(suggestions, 1):
            w(f"{i}. {sug}")

    # ---- 13. 检查点清单 ----
    w("\n## 13. 保存的检查点\n")
    ckpts = sorted([f for f in os.listdir(log_dir) if f.startswith('checkpoint_') and f.endswith('.pt')])
    if ckpts:
        for ck in ckpts:
            size_mb = os.path.getsize(os.path.join(log_dir, ck)) / 1024 / 1024
            w(f"- `{ck}` ({size_mb:.1f} MB)")
    else:
        w("(无检查点文件)")

    w("\n---")
    w("*报告由 `python -m mambadiff.analyze` 自动生成*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MambaDiff-ECR训练分析报告')
    parser.add_argument('--log_dir', type=str, default='results/mambadiff',
                        help='训练输出目录')
    parser.add_argument('--output', type=str, default=None,
                        help='报告输出路径 (默认: {log_dir}/analysis_report.md)')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"错误: 目录不存在: {args.log_dir}")
        sys.exit(1)

    report = generate_report(args.log_dir)

    output_path = args.output or os.path.join(args.log_dir, 'analysis_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"分析报告已生成: {output_path}")
    print(f"大小: {len(report)} 字符")

    # Print summary to stdout
    issues, _ = run_diagnostics(
        parse_csv(os.path.join(args.log_dir, 'phase1_metrics.csv')),
        parse_csv(os.path.join(args.log_dir, 'phase2_metrics.csv')),
        parse_log(os.path.join(args.log_dir, 'log.txt')),
        parse_config(os.path.join(args.log_dir, 'config_snapshot.yaml')),
    )
    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 未发现明显问题")


if __name__ == '__main__':
    main()
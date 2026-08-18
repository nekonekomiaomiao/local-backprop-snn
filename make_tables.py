import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else 'mnist_table_results.csv')))
# subset shared KAPPA-era rows optionally; default all
if len(sys.argv) > 2 and sys.argv[2] == 'shared_only':
    rows = [r for r in rows if r['which'] == 'shared']

def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x

print('## 整体（loss 平台 / 冻结 acc / 全局对齐）\n')
print('| which | seed | N | ΔT | R | α | κ | loss 平台 | 冻结 acc | align_all |')
print('|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['which']} | {r['seed']} | {r['N']} | {f(r['SAMPLE_T'],1)} | {f(r['R'],0)} | {r['alpha']} | {f(r['KAPPA'],2)} | {f(r['loss_plateau'])}±{f(r['loss_std'])} | {f(r['frozen_acc'])} | {f(r['align_all'],3)}±{f(r['align_std'])} |")

print('\n## 梯度夹角（逐样本 cos(ΔP,−g)，±std）\n')
print('| which | α | κ | W1 | W2 | W3 | 全局 |')
print('|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['which']} | {r['alpha']} | {f(r['KAPPA'],2)} | {f(r['align_W1'],3)} | {f(r['align_W2'],3)} | {f(r['align_W3'],3)} | {f(r['align_all'],3)} |")

print('\n## 更新方差 / SNR / 期望更新效率 / 符号一致率\n')
print('| which | α | κ | SNR W1/W2/W3 | expEff W1/W2/W3 | signCons W1/W2/W3 |')
print('|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['which']} | {r['alpha']} | {f(r['KAPPA'],2)} | {f(r['snr_W1'])}/{f(r['snr_W2'])}/{f(r['snr_W3'])} | {f(r['eff_W1'])}/{f(r['eff_W2'])}/{f(r['eff_W3'])} | {f(r['sign_W1'])}/{f(r['sign_W2'])}/{f(r['sign_W3'])} |")

print('\n## 协方差（corr(e,d) 与 |cov|/|E[e]E[d]|）\n')
print('| which | α | κ | corr W1/W2/W3 | bias W1/W2/W3 |')
print('|---|---|---|---|---|')
for r in rows:
    print(f"| {r['which']} | {r['alpha']} | {f(r['KAPPA'],2)} | {f(r['corr_W1'],3)}/{f(r['corr_W2'],3)}/{f(r['corr_W3'],3)} | {f(r['bias_W1'])}/{f(r['bias_W2'])}/{f(r['bias_W3'])} |")

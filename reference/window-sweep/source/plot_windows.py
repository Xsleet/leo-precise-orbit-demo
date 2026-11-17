"""读取已经算好的结果，画输入长度对比和历史误差到未来预报的图。"""
import argparse
import csv
import numpy as np
from main import ROOT
from short_2019 import DATA_PATH
from no_warmup import FIT_HOURS, CONTEXT_HOURS, history, prepare, future_reference
from rdop import to_swr
from plotting import plt


def plot(folder):
    with (folder/'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    scores = {(int(r['fit_hours']), int(r['context_hours'])): r for r in rows}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for hours, color in zip(FIT_HOURS, ['#147d78', '#bd3764', '#6d5cae']):
        values = [100*float(scores[hours, c]['endpoint_rmse_3d_m']) for c in CONTEXT_HOURS]
        ax.plot(CONTEXT_HOURS, values, 'o-', color=color, label=f'{hours} h fit + TimesFM')
        ax.axhline(100*float(scores[hours, 0]['endpoint_rmse_3d_m']), color=color, ls='--',
                   alpha=.65, label=f'{hours} h fit, dynamics only')
    ax.set(xlabel='TimesFM input length (h)', ylabel='60-min endpoint 3-D RMSE (cm)',
           xticks=CONTEXT_HOURS, title='No orbit warm-up | all eight fixed 2019 windows')
    ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(folder/'comparison.png', dpi=160); plt.close(fig)

    # 全表中的最小末端 RMSE，仅作示例；不把筛选后的最优值当成独立验证。
    best = min((r for r in rows if int(r['context_hours'])>0), key=lambda r:float(r['endpoint_rmse_3d_m']))
    fit_hours, context = int(best['fit_hours']), int(best['context_hours'])
    data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
    arc = prepare(history(data, 1, fit_hours))
    reference = future_reference(data, 1)[:, 1:4]
    predictions = np.genfromtxt(folder/'prediction.csv', delimiter=',', names=True)
    selected = predictions[(predictions['fit_hours']==fit_hours) &
                           (predictions['context_hours']==context) & (predictions['window']==1)]
    predicted = np.column_stack([selected[f'predicted_{c}_m'] for c in 'xyz'])
    actual_error = to_swr(reference-arc['baseline'], arc['future_basis'])
    forecast_error = to_swr(predicted-arc['baseline'], arc['future_basis'])
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), gridspec_kw={'height_ratios':[1, 1, 1, 1.2]})
    for j, label in enumerate(['Along-track S', 'Cross-track W', 'Radial R']):
        ax = axes[j]
        ax.plot(range(-360, 1), arc['residual'][-361:, j]*100, color='#44536b', label='Past fit residual')
        ax.plot(range(1, 61), actual_error[:, j]*100, 'k:', label='Future reference error (scoring only)')
        ax.plot(range(1, 61), forecast_error[:, j]*100, color='#bd3764', label='TimesFM error forecast')
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.axvspan(0, 60, color='#f5d0a2', alpha=.25)
        ax.set(xlim=(-360, 60), ylabel=label+' (cm)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=3)
    axes[2].set_xlabel('Minutes relative to forecast start | past 6 h shown -> future 60 min')
    for xyz, label, color in [(arc['baseline'], 'Dynamics', '#147d78'),
                              (predicted, 'Dynamics + TimesFM', '#bd3764')]:
        axes[3].plot(range(1, 61), 100*np.linalg.norm(xyz-reference, axis=1), color=color, label=label)
    axes[3].set(xlim=(1, 60), xlabel='Forecast lead (min)', ylabel='3-D position error (cm)')
    axes[3].grid(alpha=.2); axes[3].legend(fontsize=8)
    fig.suptitle(f'GRACE-FO C | 2019-01-02 00:00 GPS | {fit_hours} h fit, {context} h input, no orbit warm-up\n'
                 'Illustration: lowest endpoint RMSE configuration in this sweep; original TimesFM weights', fontsize=11)
    fig.tight_layout(); fig.savefig(folder/'example.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=lambda p:ROOT/p, default=ROOT/'outputs/window-sweep-timesfm-gpu')
    plot(parser.parse_args().folder)

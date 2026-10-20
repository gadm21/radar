"""Generate icc_assets/roc.tex: ROC curves (minute level) for Fig. 8.

Reads test_probs.npz written by `occupancy_pipeline.py models --probs`.
Per (window, model, features): probabilities averaged over seeds (the
balanced test cohort is identical across seeds), then sklearn's
roc_curve on the mean probabilities. Classical models see up to 3
seeds; encoder-decoder models one seed.

Style: color = input modality group (CSI=blue, radar=orange,
fusion=red), line style cycles per combo inside the group.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import occupancy_pipeline as P          # noqa: E402

OUT = P.OUT
ASSETS = Path(__file__).resolve().parents[1] / 'icc_assets'

# group: (group name, pgf color)
GROUPS = {'csi': ('CSI', 'valColor'),
          'radar': ('radar', 'testColor'),
          'fusion': ('fusion', 'occColor'),
          'moe': ('fusion', 'occColor'),
          'dual': ('fusion', 'occColor')}
STYLES = ['solid', 'densely dashed']      # up to two combos per group
MAX_PTS = 64
KEEP_PER_GROUP = 2                        # best curves per modality group


def _group(model, features):
    if features == 'dl':
        return model.split('-')[0]          # csi-* / radar-* / dual-*
    return features


def _label(model, features):
    mdl = {'rf': 'RF', 'xgb': 'XGB'}.get(model, model.upper())
    grp = GROUPS[_group(model, features)][0]
    if features == 'dl':
        return mdl.replace('-', '\\,')
    suf = {'csi': 'CSI', 'radar': 'radar',
           'fusion': 'cat', 'moe': 'MoE'}[features]
    return f'{mdl}\\,{suf}'


def main():
    z = np.load(OUT / 'test_probs.npz')
    store = {}
    for key in z.files:
        w, m, f, s = key.split('|')
        y, p = z[key]
        store.setdefault((int(w), m, f), []).append((int(s), y, p))

    panels = {w: [] for w in P.WIN_LIST}
    for (w, m, f), seeds in store.items():
        if w not in panels:
            continue
        seeds.sort()
        y = seeds[0][1]
        pmean = np.mean([p for _, _, p in seeds], axis=0)
        fpr, tpr, _ = roc_curve(y, pmean)
        auc = roc_auc_score(y, pmean)
        if len(fpr) > MAX_PTS:
            idx = np.unique(np.linspace(0, len(fpr) - 1, MAX_PTS)
                            .round().astype(int))
            fpr, tpr = fpr[idx], tpr[idx]
        panels[w].append((_group(m, f), m, f, _label(m, f), auc,
                          fpr, tpr))

    # keep the best KEEP_PER_GROUP combos per modality group, selected
    # globally by mean AUC across windows so both panels show the same
    # curves under one shared legend
    score = {}
    for w, combos in panels.items():
        for c in combos:
            # key by modality group name so cat/MoE/dual share 'fusion'
            score.setdefault((GROUPS[c[0]][0], c[0], c[1], c[2]),
                             []).append(c[4])
    keep = set()
    for gname in {k[0] for k in score}:
        g = sorted((k for k in score if k[0] == gname),
                   key=lambda k: -np.mean(score[k]))
        keep.update(g[:KEEP_PER_GROUP])
    for w in panels:
        panels[w] = [c for c in panels[w]
                     if (GROUPS[c[0]][0], c[0], c[1], c[2]) in keep]

    rank = {'csi': 0, 'radar': 1, 'fusion': 2, 'moe': 3, 'dual': 4}
    mrank = {'rf': 0, 'xgb': 1, 'cnn': 2, 'lstm': 3}
    s = ['% generated ROC curves (minute level)',
         '\\begin{tikzpicture}',
         '\\begin{groupplot}[group style={group size=2 by 1,'
         'horizontal sep=1.4cm},',
         '  width=0.40\\textwidth, height=5.2cm,',
         '  xmin=0, xmax=1, ymin=0, ymax=1,',
         '  xlabel={false positive rate}, ylabel={true positive rate},',
         '  tick label style={font=\\scriptsize},',
         '  label style={font=\\scriptsize}]']

    for pi, w in enumerate(P.WIN_LIST):
        extra = (', legend to name=rocleg, legend columns=6,'
                 'legend style={font=\\scriptsize,draw=none,fill=none,'
                 'column sep=8pt}' if pi == 0 else '')
        s.append(f'\\nextgroupplot[title={{\\scriptsize $W={w}$\\,s}}'
                 f'{extra}]')
        s.append('\\addplot[gray!60,dashed,forget plot] coordinates '
                 '{(0,0) (1,1)};')
        combos = sorted(panels[w],
                        key=lambda c: (rank.get(c[0], 9),
                                       mrank.get(c[1].split('-')[-1], 9),
                                       c[2], c[1]))
        ctr = {}
        for grp, m, f, lbl, auc, fpr, tpr in combos:
            i = ctr.get(grp, 0)
            ctr[grp] = i + 1
            color = GROUPS[grp][1]
            style = ('solid' if i == 0
                     else 'densely dashed')
            colr = color if i == 0 else color + '!60!black'
            pts = ' '.join(f'({a:.3f},{b:.3f})' for a, b in zip(fpr, tpr))
            s.append(f'\\addplot[{colr},{style},very thick] '
                     f'coordinates {{{pts}}};')
            if pi == 0:
                s.append(f'\\addlegendentry{{{lbl}}}')

    s += ['\\end{groupplot}', '\\end{tikzpicture}',
          '', '\\par\\vspace{4pt}\\ref{rocleg}']

    ASSETS.mkdir(exist_ok=True)
    (ASSETS / 'roc.tex').write_text('\n'.join(s) + '\n')
    print('wrote', ASSETS / 'roc.tex')


if __name__ == '__main__':
    main()

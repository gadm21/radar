#!/usr/bin/env python3
"""Evaluate the saved dual-head activity model and serve an interactive web report.

Usage example:
    python test.py minutes --output_dir output --max_per_class 500 --port 8000
"""

import argparse
import inspect
import json
import math
import signal
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve


def _tsne_iter_kwarg(value=500):
    if "max_iter" in inspect.signature(TSNE.__init__).parameters:
        return {"max_iter": value}
    return {"n_iter": value}


from ml_utils import (
    DualHeadActivityModel,
    PreprocessingConfig,
    build_balanced_chunk_arrays,
    compute_chunk_stats,
    compute_multiclass_metrics,
    discover_chunk_records,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", default="minutes", nargs="?", help="Root folder containing per-minute subdirectories")
    parser.add_argument("--output_dir", default="output", help="Directory with saved model and outputs")
    parser.add_argument("--max_per_class", type=int, default=500, help="Max test chunks per class (0 = all)")
    parser.add_argument("--balance_test", action="store_true", help="Balance test classes by oversampling")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--port", type=int, default=8000, help="Port for the dashboard web server")
    parser.add_argument("--n_dashboard_samples", type=int, default=12, help="Number of sample cards in the explorer")
    parser.add_argument("--no_server", action="store_true", help="Generate dashboard and results, then exit without serving")
    return parser.parse_args()


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def sanitize_json(obj):
    """Convert NaN to None and numpy arrays to lists for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def load_outputs(output_dir):
    output_dir = Path(output_dir)
    model_path = output_dir / "activity_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    model, prep_cfg = load_model(model_path)
    cfg = PreprocessingConfig(**prep_cfg)
    with open(output_dir / "label_to_idx.json", "r", encoding="utf-8") as handle:
        label_to_idx = json.load(handle)
    return model, cfg, label_to_idx


def build_test_data(args, cfg):
    records, label_to_idx = discover_chunk_records(args.root, cfg)
    stats = compute_chunk_stats(records)
    max_test = args.max_per_class if args.max_per_class > 0 else None
    test_radar, test_csi, test_labels, _ = build_balanced_chunk_arrays(
        records, cfg, split="test", max_per_class=max_test, balance=args.balance_test, random_state=42
    )
    return records, stats, test_radar, test_csi, test_labels, label_to_idx


def run_inference(model, radar, csi, labels, device, batch_size=32):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(radar), torch.from_numpy(csi), torch.from_numpy(labels)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    all_y, all_prob = [], []
    all_radar, all_csi, all_fused, all_attn = [], [], [], []
    with torch.no_grad():
        for r, c, y in loader:
            r = r.to(device)
            c = c.to(device)
            _, _, fused, attn = model.get_embeddings(r, c)
            logits = model.classifier(fused)
            prob = F.softmax(logits, dim=1)
            all_y.extend(y.numpy())
            all_prob.extend(prob.cpu().numpy())
            all_radar.extend(r.cpu().numpy())
            all_csi.extend(c.cpu().numpy())
            all_fused.extend(fused.cpu().numpy())
            all_attn.extend(attn.cpu().numpy())
    all_y = np.asarray(all_y, dtype=int)
    all_prob = np.asarray(all_prob, dtype=float)
    all_radar = np.asarray(all_radar, dtype=float)
    all_csi = np.asarray(all_csi, dtype=float)
    all_fused = np.asarray(all_fused, dtype=float)
    all_attn = np.asarray(all_attn, dtype=float)
    return all_y, all_prob, all_radar, all_csi, all_fused, all_attn


def compute_roc_curves(y_true, y_prob, labels):
    rocs = []
    for i, name in enumerate(labels):
        yt = (y_true == i).astype(int)
        if len(np.unique(yt)) < 2:
            rocs.append({"class": name, "fpr": [0, 1], "tpr": [0, 1], "auc": None})
            continue
        fpr, tpr, _ = roc_curve(yt, y_prob[:, i])
        # downsample for smaller JSON
        if len(fpr) > 300:
            idx = np.linspace(0, len(fpr) - 1, 300, dtype=int)
            fpr, tpr = fpr[idx], tpr[idx]
        auc = float(roc_auc_score(yt, y_prob[:, i]))
        rocs.append({"class": name, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc})
    return rocs


def build_dashboard_data(args, cfg, model, records, stats, radar, csi, labels, label_to_idx):
    n_classes = len(label_to_idx)
    class_names = [name for name, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    y_true, y_prob, _, _, fused, attn = run_inference(model, radar, csi, labels, torch.device(args.device), batch_size=32)
    metrics = compute_multiclass_metrics(y_true, y_prob, n_classes, class_names)
    rocs = compute_roc_curves(y_true, y_prob, class_names)

    confidences = y_prob.max(axis=1)
    correct_mask = (y_prob.argmax(axis=1) == y_true)

    # Embeddings
    if len(fused) < 2:
        pca = np.zeros((len(fused), 2), dtype=float)
        tsne = np.zeros((len(fused), 2), dtype=float)
        tsne_yt = y_true
        tsne_yp = y_prob.argmax(axis=1)
    else:
        pca = PCA(n_components=2, random_state=42).fit_transform(fused)
        if len(fused) > 2000:
            tsne_idx = np.random.RandomState(42).choice(len(fused), 1000, replace=False)
            tsne_fused = fused[tsne_idx]
            tsne_yt = y_true[tsne_idx]
            tsne_yp = y_prob.argmax(axis=1)[tsne_idx]
        else:
            tsne_fused = fused
            tsne_yt = y_true
            tsne_yp = y_prob.argmax(axis=1)
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(1, len(tsne_fused) - 1)),
            random_state=42,
            **_tsne_iter_kwarg(500),
        ).fit_transform(tsne_fused)

    # Sample explorer
    n_samples = args.n_dashboard_samples
    n_per_class = max(1, n_samples // n_classes)
    selected = []
    rng = np.random.RandomState(42)
    for i, name in enumerate(class_names):
        idx = np.where(y_true == i)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        selected.extend(chosen.tolist())
    selected = selected[:n_samples]

    frame_idx = cfg.radar_frames // 2
    samples = []
    for idx in selected:
        s = {
            "true_label": class_names[y_true[idx]],
            "pred_label": class_names[y_prob.argmax(axis=1)[idx]],
            "confidence": float(confidences[idx]),
            "probs": y_prob[idx].tolist(),
            "radar_maps": [radar[idx, frame_idx, :, :, c].tolist() for c in range(radar.shape[-1])],
            "csi": csi[idx, :, :].tolist(),
            "attention": attn[idx, :, :].tolist(),
        }
        samples.append(s)

    # Training curves
    curves_path = Path(args.output_dir) / "activity_training_stats.csv"
    training_curves = []
    if curves_path.exists():
        training_curves = pd.read_csv(curves_path).to_dict(orient="records")

    model_summary = {
        "total_params": int(count_parameters(model)),
        "radar_params": int(count_parameters(model.radar_encoder)),
        "csi_params": int(count_parameters(model.csi_encoder)),
        "attention_params": int(count_parameters(model.attention) + count_parameters(model.norm)),
        "classifier_params": int(count_parameters(model.classifier)),
        "architecture": str(model),
    }

    data = {
        "labels": class_names,
        "label_to_idx": label_to_idx,
        "n_classes": n_classes,
        "class_distribution": stats["by_split"],
        "total_chunks": stats["total_chunks"],
        "n_test": len(y_true),
        "model_summary": model_summary,
        "training_curves": training_curves,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class": metrics["per_class"],
        },
        "roc_curves": rocs,
        "confidences": {
            "all": confidences.tolist(),
            "correct": confidences[correct_mask].tolist(),
            "incorrect": confidences[~correct_mask].tolist(),
        },
        "embeddings": {
            "pca": {"x": pca[:, 0].tolist(), "y": pca[:, 1].tolist(), "true": y_true.tolist(), "pred": y_prob.argmax(axis=1).tolist()},
            "tsne": {"x": tsne[:, 0].tolist(), "y": tsne[:, 1].tolist(), "true": tsne_yt.tolist(), "pred": tsne_yp.tolist()},
        },
        "samples": samples,
    }
    return data, metrics


def write_dashboard_html(data, html_path):
    json_text = json.dumps(sanitize_json(data), allow_nan=False, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dual-Head Radar+CSI Activity Classifier — Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f5f7fa; color: #1a202c; margin: 0; padding: 20px; }
  h1 { font-size: 1.8rem; margin-bottom: 0.3rem; }
  h2 { font-size: 1.4rem; margin-top: 1.5rem; margin-bottom: 0.8rem; color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem; }
  h3 { font-size: 1.1rem; color: #4a5568; }
  .subtitle { color: #718096; margin-bottom: 1.5rem; }
  .section { background: #fff; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }
  .chart { min-height: 340px; }
  .half { min-height: 320px; }
  .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; }
  .card { background: #edf2f7; border-radius: 6px; padding: 0.8rem; text-align: center; }
  .card .big { font-size: 1.5rem; font-weight: 700; color: #2b6cb0; }
  .card .label { font-size: 0.85rem; color: #4a5568; margin-top: 0.2rem; }
  pre.arch { background: #2d3748; color: #e2e8f0; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; max-height: 300px; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { border: 1px solid #e2e8f0; padding: 0.5rem; text-align: left; }
  th { background: #edf2f7; }
  .sample-controls { margin-bottom: 0.8rem; }
  .sample-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
  .sample-plot { min-height: 220px; }
  select { padding: 0.4rem 0.8rem; font-size: 1rem; border: 1px solid #cbd5e0; border-radius: 4px; background: #fff; }
  .pred-text { margin: 0.6rem 0; font-size: 0.95rem; }
</style>
</head>
<body>
<h1>Dual-Head Radar + CSI Activity Classifier</h1>
<div class="subtitle">Interactive test report generated by <code>test.py</code></div>

<div class="section">
  <h2>Data Overview</h2>
  <p>Total chunks discovered: <strong id="total-chunks"></strong> &nbsp;|&nbsp; Test chunks: <strong id="test-chunks"></strong></p>
  <div id="data-overview-chart" class="chart"></div>
  <table id="class-table"><thead><tr><th>Split</th><th id="class-th">Activity</th><th>Chunks</th></tr></thead><tbody></tbody></table>
</div>

<div class="section">
  <h2>Model Architecture</h2>
  <div class="summary" id="model-summary"></div>
  <h3>PyTorch module summary</h3>
  <pre class="arch" id="architecture"></pre>
</div>

<div class="section">
  <h2>Training Curves</h2>
  <div class="grid">
    <div id="curve-loss" class="chart"></div>
    <div id="curve-f1" class="chart"></div>
  </div>
</div>

<div class="section">
  <h2>Test Metrics</h2>
  <div class="summary" id="test-summary"></div>
  <div class="grid">
    <div id="confusion-matrix" class="chart"></div>
    <div id="per-class-metrics" class="chart"></div>
  </div>
</div>

<div class="section">
  <h2>ROC Curves & Confidence</h2>
  <div class="grid">
    <div id="roc-curves" class="chart"></div>
    <div id="confidence-hist" class="chart"></div>
  </div>
</div>

<div class="section">
  <h2>Embedding Visualizations</h2>
  <div class="grid">
    <div id="embed-pca-true" class="half"></div>
    <div id="embed-tsne-true" class="half"></div>
    <div id="embed-pca-pred" class="half"></div>
    <div id="embed-tsne-pred" class="half"></div>
  </div>
</div>

<div class="section">
  <h2>Sample Explorer</h2>
  <div class="sample-controls">
    <label for="sample-select">Select sample: </label>
    <select id="sample-select"></select>
  </div>
  <div class="pred-text" id="sample-pred"></div>
  <div class="sample-grid">
    <div id="sample-rd" class="sample-plot"></div>
    <div id="sample-ar" class="sample-plot"></div>
    <div id="sample-ad" class="sample-plot"></div>
    <div id="sample-csi" class="sample-plot"></div>
    <div id="sample-attn" class="sample-plot"></div>
  </div>
</div>

<script type="application/json" id="dashboard-data">{{DATA}}</script>
<script>
const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);

function createDataOverview() {
  document.getElementById('total-chunks').textContent = DATA.total_chunks;
  document.getElementById('test-chunks').textContent = DATA.n_test;
  const splits = Object.keys(DATA.class_distribution);
  const labels = DATA.labels;
  const traces = splits.map(split => ({
    x: labels,
    y: labels.map(l => DATA.class_distribution[split][l] || 0),
    name: split,
    type: 'bar'
  }));
  Plotly.newPlot('data-overview-chart', traces, {
    barmode: 'group',
    title: 'Class distribution by split',
    xaxis: { title: 'Activity' },
    yaxis: { title: 'Chunks' },
    margin: { t: 40 }
  });

  const tbody = document.querySelector('#class-table tbody');
  splits.forEach(split => {
    labels.forEach(label => {
      const count = DATA.class_distribution[split][label] || 0;
      const row = document.createElement('tr');
      row.innerHTML = `<td>${split}</td><td>${label}</td><td>${count}</td>`;
      tbody.appendChild(row);
    });
  });
}

function createModelSummary() {
  const s = DATA.model_summary;
  const container = document.getElementById('model-summary');
  const cards = [
    { label: 'Total parameters', value: s.total_params.toLocaleString() },
    { label: 'Radar encoder params', value: s.radar_params.toLocaleString() },
    { label: 'CSI encoder params', value: s.csi_params.toLocaleString() },
    { label: 'Attention params', value: s.attention_params.toLocaleString() },
    { label: 'Classifier params', value: s.classifier_params.toLocaleString() },
  ];
  cards.forEach(c => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `<div class="big">${c.value}</div><div class="label">${c.label}</div>`;
    container.appendChild(div);
  });
  document.getElementById('architecture').textContent = s.architecture;
}

function createTrainingCurves() {
  if (!DATA.training_curves || DATA.training_curves.length === 0) {
    ['curve-loss', 'curve-f1'].forEach(id => document.getElementById(id).innerHTML = '<p>No training stats available.</p>');
    return;
  }
  const epochs = DATA.training_curves.map(r => r.epoch);
  const lossTr = DATA.training_curves.map(r => r.train_loss);
  const lossTe = DATA.training_curves.map(r => r.test_loss);
  const f1Tr = DATA.training_curves.map(r => r.train_macro_f1);
  const f1Te = DATA.training_curves.map(r => r.test_macro_f1);

  Plotly.newPlot('curve-loss', [
    { x: epochs, y: lossTr, name: 'Train loss', mode: 'lines' },
    { x: epochs, y: lossTe, name: 'Test loss', mode: 'lines' }
  ], { title: 'Loss', xaxis: { title: 'Epoch' }, yaxis: { title: 'Loss' }, margin: { t: 40 } });

  Plotly.newPlot('curve-f1', [
    { x: epochs, y: f1Tr, name: 'Train macro-F1', mode: 'lines' },
    { x: epochs, y: f1Te, name: 'Test macro-F1', mode: 'lines' }
  ], { title: 'Macro F1', xaxis: { title: 'Epoch' }, yaxis: { title: 'Macro F1', range: [0, 1] }, margin: { t: 40 } });
}

function createTestMetrics() {
  const m = DATA.metrics;
  const container = document.getElementById('test-summary');
  [
    { label: 'Accuracy', value: (m.accuracy * 100).toFixed(2) + '%' },
    { label: 'Macro F1', value: m.macro_f1.toFixed(3) },
    { label: 'Macro Precision', value: m.macro_precision.toFixed(3) },
    { label: 'Macro Recall', value: m.macro_recall.toFixed(3) },
  ].forEach(c => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `<div class="big">${c.value}</div><div class="label">${c.label}</div>`;
    container.appendChild(div);
  });

  const labels = DATA.labels;
  Plotly.newPlot('confusion-matrix', [{
    z: m.confusion_matrix,
    x: labels,
    y: labels,
    type: 'heatmap',
    colorscale: 'Blues',
    hoverongaps: false,
    reversescale: false
  }], { title: 'Confusion matrix', xaxis: { title: 'Predicted' }, yaxis: { title: 'True' }, margin: { t: 40 } });

  const classes = Object.keys(m.per_class);
  const p = classes.map(c => m.per_class[c].precision);
  const r = classes.map(c => m.per_class[c].recall);
  const f = classes.map(c => m.per_class[c].f1);
  Plotly.newPlot('per-class-metrics', [
    { x: classes, y: p, name: 'Precision', type: 'bar' },
    { x: classes, y: r, name: 'Recall', type: 'bar' },
    { x: classes, y: f, name: 'F1', type: 'bar' }
  ], { barmode: 'group', title: 'Per-class metrics', yaxis: { range: [0, 1] }, margin: { t: 40 } });
}

function createROCCurves() {
  const traces = DATA.roc_curves.map((roc, i) => ({
    x: roc.fpr,
    y: roc.tpr,
    name: `${roc.class} (AUC ${roc.auc === null ? 'N/A' : roc.auc.toFixed(3)})`,
    mode: 'lines'
  }));
  traces.push({ x: [0, 1], y: [0, 1], name: 'Random', mode: 'lines', line: { dash: 'dash', color: 'gray' } });
  Plotly.newPlot('roc-curves', traces, {
    title: 'One-vs-rest ROC curves',
    xaxis: { title: 'False positive rate' },
    yaxis: { title: 'True positive rate' },
    margin: { t: 40 }
  });

  Plotly.newPlot('confidence-hist', [
    { x: DATA.confidences.correct, name: 'Correct', type: 'histogram', opacity: 0.7, marker: { color: 'green' } },
    { x: DATA.confidences.incorrect, name: 'Incorrect', type: 'histogram', opacity: 0.7, marker: { color: 'red' } }
  ], {
    barmode: 'overlay',
    title: 'Prediction confidence',
    xaxis: { title: 'Max predicted probability', range: [0, 1] },
    yaxis: { title: 'Count' },
    margin: { t: 40 }
  });
}

function makeScatter(id, emb, colorBy, title) {
  const labels = DATA.labels;
  const traces = labels.map((lab, i) => ({
    x: emb.x.filter((_, j) => colorBy[j] === i),
    y: emb.y.filter((_, j) => colorBy[j] === i),
    name: lab,
    mode: 'markers',
    type: 'scatter',
    marker: { size: 6, opacity: 0.7 }
  }));
  Plotly.newPlot(id, traces, {
    title: title,
    xaxis: { title: 'Component 1', showgrid: false, zeroline: false },
    yaxis: { title: 'Component 2', showgrid: false, zeroline: false },
    margin: { t: 40 }
  });
}

function createEmbeddings() {
  makeScatter('embed-pca-true', DATA.embeddings.pca, DATA.embeddings.pca.true, 'PCA (true labels)');
  makeScatter('embed-pca-pred', DATA.embeddings.pca, DATA.embeddings.pca.pred, 'PCA (predicted labels)');
  makeScatter('embed-tsne-true', DATA.embeddings.tsne, DATA.embeddings.tsne.true, 't-SNE (true labels)');
  makeScatter('embed-tsne-pred', DATA.embeddings.tsne, DATA.embeddings.tsne.pred, 't-SNE (predicted labels)');
}

function renderSample(idx) {
  const s = DATA.samples[idx];
  const probs = s.probs.map((p, i) => `${DATA.labels[i]}: ${p.toFixed(3)}`).join(' &nbsp;|&nbsp; ');
  document.getElementById('sample-pred').innerHTML =
    `<strong>True:</strong> ${s.true_label} &nbsp;|&nbsp; <strong>Pred:</strong> ${s.pred_label} &nbsp;|&nbsp; <strong>Confidence:</strong> ${s.confidence.toFixed(3)}<br>Probabilities: ${probs}`;

  const commonLayout = { margin: { t: 30, l: 30, r: 20, b: 30 }, xaxis: { showticklabels: false }, yaxis: { showticklabels: false } };
  Plotly.newPlot('sample-rd', [{ z: s.radar_maps[0], type: 'heatmap', colorscale: 'Viridis', showscale: false }], { ...commonLayout, title: 'RD' });
  Plotly.newPlot('sample-ar', [{ z: s.radar_maps[1], type: 'heatmap', colorscale: 'Viridis', showscale: false }], { ...commonLayout, title: 'AR' });
  Plotly.newPlot('sample-ad', [{ z: s.radar_maps[2], type: 'heatmap', colorscale: 'Viridis', showscale: false }], { ...commonLayout, title: 'AD' });
  Plotly.newPlot('sample-csi', [{ z: s.csi, type: 'heatmap', colorscale: 'Plasma', showscale: false }], { ...commonLayout, title: 'CSI amplitude' });
  Plotly.newPlot('sample-attn', [{ z: s.attention, x: ['radar', 'csi'], y: ['radar', 'csi'], type: 'heatmap', colorscale: 'Blues', showscale: false }], { ...commonLayout, title: 'Attention (2 tokens)' });
}

function createSampleExplorer() {
  const select = document.getElementById('sample-select');
  DATA.samples.forEach((s, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Sample ${i + 1}: true=${s.true_label}, pred=${s.pred_label}`;
    select.appendChild(opt);
  });
  select.addEventListener('change', () => renderSample(parseInt(select.value, 10)));
  renderSample(0);
}

function buildDashboard() {
  createDataOverview();
  createModelSummary();
  createTrainingCurves();
  createTestMetrics();
  createROCCurves();
  createEmbeddings();
  createSampleExplorer();
}

buildDashboard();
</script>
</body>
</html>"""
    html = html.replace("{{DATA}}", json_text)
    Path(html_path).write_text(html, encoding="utf-8")


def write_test_results(metrics, output_dir):
    with open(Path(output_dir) / "test_results.json", "w", encoding="utf-8") as handle:
        json.dump(sanitize_json(metrics), handle, indent=2, ensure_ascii=False)


def start_server(output_dir, port):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=output_dir, **kwargs)

    server = ThreadingHTTPServer(("", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://localhost:{port}/dashboard.html"
    print(f"Dashboard server running at {url}")
    webbrowser.open(url)
    stop = threading.Event()
    def on_signal(*_):
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    print("Press Ctrl+C to stop the server...")
    try:
        while not stop.is_set():
            stop.wait(1)
    finally:
        print("\nShutting down server...")
        server.shutdown()


def main():
    args = parse_args()
    set_seed(42)

    print("Loading model and config...")
    model, cfg, label_to_idx = load_outputs(args.output_dir)
    print(f"Classes: {label_to_idx}")

    print("Discovering and loading test chunks...")
    records, stats, radar, csi, labels, _ = build_test_data(args, cfg)
    print(f"Test chunks loaded: {len(labels)}")

    print("Building dashboard...")
    dashboard_data, metrics = build_dashboard_data(args, cfg, model, records, stats, radar, csi, labels, label_to_idx)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_dashboard_html(dashboard_data, output_dir / "dashboard.html")
    print(f"Saved dashboard to {output_dir / 'dashboard.html'}")

    write_test_results(metrics, output_dir)
    print(f"Saved test results to {output_dir / 'test_results.json'}")

    if args.no_server:
        print("--no_server set; skipping HTTP server.")
        return

    start_server(output_dir, args.port)


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PAL_REF = "#4878CF"
PAL_PROD = "#D65F5F"


def _plot_feature_hists(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
    features: list[str],
    out_path: Path,
    title: str,
) -> None:
    n = len(features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = [axes] if n == 1 else axes.flatten()
    for ax, feat in zip(axes, features):
        ref_vals = df_ref[feat].dropna().values
        prod_vals = df_prod[feat].dropna().values
        ax.hist(ref_vals, bins=40, alpha=0.6, color=PAL_REF, label="Referencia", density=True)
        ax.hist(prod_vals, bins=40, alpha=0.6, color=PAL_PROD, label="Actual", density=True)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_importances(importances: dict, out_path: Path, title: str) -> None:
    items = sorted(importances.items(), key=lambda x: x[1])
    feats, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(feats))))
    ax.barh(feats, vals, color="#4C72B0")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("Importancia (RandomForest ref-vs-prod)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _ks_table_html(per_feature: list[dict]) -> str:
    rows = []
    for f in per_feature:
        cls = "drift" if f["drift"] else "ok"
        rows.append(
            f"<tr class='{cls}'><td>{f['feature']}</td>"
            f"<td>{f['D_KS']:.4f}</td>"
            f"<td>{f['p_value']:.4f}</td>"
            f"<td>{'DRIFT' if f['drift'] else 'ok'}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Feature</th><th>D_KS</th>"
        "<th>p-value</th><th>Estado</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _decay_table_html(decay: dict) -> str:
    return (
        "<table><thead><tr><th>Métrica</th><th>Entrenamiento</th>"
        "<th>Actual</th><th>Δ</th></tr></thead><tbody>"
        f"<tr><td>MSE</td><td>{decay['mse_train']:.4f}</td>"
        f"<td>{decay['mse_current']:.4f}</td><td>{decay['mse_delta']:+.4f}</td></tr>"
        f"<tr><td>R²</td><td>{decay['r2_train']:.4f}</td>"
        f"<td>{decay['r2_current']:.4f}</td><td>{decay['r2_delta']:+.4f}</td></tr>"
        f"<tr><td>n_samples</td><td colspan='3'>{decay['n_samples']}</td></tr>"
        "</tbody></table>"
    )


def _model_section(
    model_label: str,
    ks: dict,
    clf: dict,
    decay: dict,
    hist_img: str,
    importances_img: str,
) -> str:
    verdict_drift = ks["is_drift"] or clf["is_drift"]
    verdict_decay = abs(decay["r2_delta"]) > 0.1
    verdict = (
        "<span class='bad'>REVISAR MODELO</span>"
        if verdict_drift or verdict_decay
        else "<span class='good'>OK</span>"
    )
    return f"""
    <section>
      <h2>{model_label} <small>(target={decay['target']})</small> &mdash; {verdict}</h2>

      <h3>KS Drift (univariado)</h3>
      <p>Features con drift: <b>{ks['n_drifted']} / {len(ks['per_feature'])}</b>.
         p-value mínimo: <b>{ks['min_p_value']:.4f}</b>.</p>
      {_ks_table_html(ks['per_feature'])}

      <h3>Classifier Drift (multivariado)</h3>
      <p>AUC del clasificador ref-vs-actual: <b>{clf['auc']:.4f}</b>
         (p-value <b>{clf['p_value']:.4f}</b>).
         {"<b class='bad'>DRIFT DETECTADO</b>" if clf['is_drift'] else "<b class='good'>Sin drift</b>"}.</p>
      <img src="{importances_img}" alt="Feature importances {model_label}">

      <h3>Model decay</h3>
      {_decay_table_html(decay)}

      <h3>Distribuciones por feature</h3>
      <img src="{hist_img}" alt="Histogramas {model_label}">
    </section>
    """


HTML_HEAD = """<!doctype html>
<html><head><meta charset='utf-8'><title>Drift &amp; Decay Report</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1080px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { border-bottom: 2px solid #4878CF; padding-bottom: 0.3em; }
h2 { margin-top: 2em; }
table { border-collapse: collapse; margin: 0.8em 0; }
th, td { border: 1px solid #ddd; padding: 4px 10px; font-size: 0.9em; }
th { background: #f4f4f4; }
tr.drift td { background: #fde4e4; }
tr.ok td { background: #eaf6ea; }
img { max-width: 100%; margin: 0.6em 0; border: 1px solid #eee; }
.good { color: #1f7a3a; font-weight: bold; }
.bad { color: #b22222; font-weight: bold; }
small { color: #666; font-weight: normal; }
.meta { color: #555; font-size: 0.9em; }
</style></head><body>
"""


def generate_report(
    df_ref_gas: pd.DataFrame,
    df_ref_pet: pd.DataFrame,
    df_cur: pd.DataFrame,
    gas_results: dict,
    pet_results: dict,
    cur_snapshot: str,
    output_dir: Path,
) -> Path:
    """Genera el reporte HTML + plots + CSVs en output_dir.

    `gas_results` y `pet_results` esperan keys: `ks`, `clf`, `decay`, `features`, `ref_date`.
    Cada modelo puede tener un snapshot de referencia distinto (el del entrenamiento).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plots por modelo (cada uno contra su propio snapshot de referencia)
    _plot_feature_hists(
        df_ref_gas, df_cur, gas_results["features"],
        output_dir / "hist_gas.png",
        "gas_model: distribuciones referencia vs actual",
    )
    _plot_importances(
        gas_results["clf"]["importances"],
        output_dir / "importances_gas.png",
        "gas_model: importancia del clasificador ref-vs-actual",
    )
    _plot_feature_hists(
        df_ref_pet, df_cur, pet_results["features"],
        output_dir / "hist_pet.png",
        "pet_model: distribuciones referencia vs actual",
    )
    _plot_importances(
        pet_results["clf"]["importances"],
        output_dir / "importances_pet.png",
        "pet_model: importancia del clasificador ref-vs-actual",
    )

    # CSVs
    ks_rows = []
    for label, res in [("gas_model", gas_results), ("pet_model", pet_results)]:
        for row in res["ks"]["per_feature"]:
            ks_rows.append({"model": label, **row})
    pd.DataFrame(ks_rows).to_csv(output_dir / "ks_results.csv", index=False)

    imp_rows = []
    for label, res in [("gas_model", gas_results), ("pet_model", pet_results)]:
        for feat, imp in res["clf"]["importances"].items():
            imp_rows.append({"model": label, "feature": feat, "importance": imp})
    pd.DataFrame(imp_rows).to_csv(
        output_dir / "classifier_importances.csv", index=False,
    )

    # HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = HTML_HEAD
    html += f"<h1>Drift &amp; Decay Report</h1>"
    html += (
        "<p class='meta'>"
        f"Referencia gas: <b>{gas_results['ref_date']}</b> &mdash; "
        f"Referencia pet: <b>{pet_results['ref_date']}</b> &mdash; "
        f"Actual: <b>{cur_snapshot}</b> &mdash; "
        f"Generado: {now}"
        "</p>"
    )
    html += _model_section(
        "gas_model",
        gas_results["ks"], gas_results["clf"], gas_results["decay"],
        "hist_gas.png", "importances_gas.png",
    )
    html += _model_section(
        "pet_model",
        pet_results["ks"], pet_results["clf"], pet_results["decay"],
        "hist_pet.png", "importances_pet.png",
    )
    html += "</body></html>"

    report_path = output_dir / "drift_report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path

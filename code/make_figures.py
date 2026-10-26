"""
make_figures.py - vector figures for the paper (ACM widths: 3.33in one column, 7in two columns).

Inputs (paths below can be changed with the flags):
    --metrics   paper_tables_out/main/all_metrics.csv   (paper_tables.py; per model, seed, language)
    --diag42    diagnostics of seed 42 incl. the VLM     (modality_reliance.py --seeds 42 --vlm ...)
    --diag_x    diagnostics, XLM-R runs, 3 seeds         (modality_reliance.py on runs_xlmr)
    --diag_s    diagnostics, SigLIP-2 runs, 3 seeds      (modality_reliance.py on runs_v2)
Outputs (--out): fig_diagnostics, fig_language, fig_radar, fig_dataset  (.pdf and .png)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

TASKS = ["Crisis", "Severity", "Informativeness", "Humanitarian", "Resource"]
# Okabe-Ito colour-blind safe palette
C = {"ours": "#D55E00", "fusion": "#E69F00", "text": "#0072B2", "image": "#56B4E9", "sota": "#009E73",
     "vlm": "#CC79A7", "grey": "#7F7F7F", "dark": "#222222"}


def setup_style():
    import glob
    # register Linux Libertine (the ACM text font) explicitly; matplotlib does not always find OTF files
    for f in glob.glob("/usr/share/fonts/**/LinLibertine_R*.otf", recursive=True) + \
            glob.glob(os.path.expanduser("~/.fonts/LinLibertine_R*.otf")):
        font_manager.fontManager.addfont(f)
    fams = {f.name for f in font_manager.fontManager.ttflist}
    serif = "Linux Libertine O" if "Linux Libertine O" in fams else "DejaVu Serif"
    print("figure font:", serif)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [serif], "font.size": 7.5, "axes.titlesize": 8,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "mathtext.fontset": "stix"})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# ─────────────────────────────── Figure: step-wise diagnostics ───────────────────────────────
def fig_diagnostics(a):
    d42 = pd.read_csv(a.diag42)
    dx, ds = pd.read_csv(a.diag_x), pd.read_csv(a.diag_s)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.35, 0.8, 1.05]})

    # several VLMs: rows named "<VLM> | <setting>" (modality_reliance.py run per VLM folder, then merged);
    # older CSVs with one VLM ("VLM zero-shot (direct)", ...) are still supported
    multi = d42.model.str.contains(" \\| ", regex=True).any()
    if multi:
        vlm = d42[d42.model.str.contains(" \\| ", regex=True)].copy()
        vlm["setting"] = vlm.model.str.split(" | ", regex=False).str[1]
        vlm["group"] = vlm.setting.replace({"zero-shot direct": "zero-shot", "zero-shot CoT": "zero-shot"})
        means = vlm.groupby("group")[["acc_text_sufficient", "acc_image_dependent"]].mean()
        get = lambda key: means.loc[key] if key in means.index else d42[d42.model == key].iloc[0]
        rows = [("zero-shot", "VLMs, zero-shot", C["vlm"]), ("few-shot", "VLMs, few-shot", C["vlm"]),
                ("LoRA", "VLMs, LoRA", C["vlm"]),
                ("Fusion baseline", "Fusion", C["fusion"]), ("CCEC", "CCEC", C["ours"]), ("CCEC-G", "CCEC-G", C["ours"])]
    else:
        get = lambda key: d42[d42.model == key].iloc[0]
        rows = [("VLM zero-shot (direct)", "VLM, direct", C["vlm"]), ("VLM zero-shot (CoT)", "VLM, CoT", C["vlm"]),
                ("Fusion baseline", "Fusion", C["fusion"]), ("CCEC", "CCEC", C["ours"]), ("CCEC-G", "CCEC-G", C["ours"])]
    # (a) accuracy on text-sufficient vs image-dependent decisions, seed 42
    y = np.arange(len(rows))[::-1]
    for yi, (key, lab, col) in zip(y, rows):
        r = get(key)
        ts, idp = 100 * r.acc_text_sufficient, 100 * r.acc_image_dependent
        ax[0].plot([idp, ts], [yi, yi], color="#BBBBBB", lw=2.2, zorder=1, solid_capstyle="round")
        ax[0].scatter(idp, yi, s=26, color="white", edgecolor=col, lw=1.4, zorder=3)
        ax[0].scatter(ts, yi, s=26, color=col, zorder=3)
        ax[0].text(ts + 1.5, yi, f"{ts:.0f}", va="center", fontsize=6.3)
        ax[0].text(idp - 1.5, yi, f"{idp:.0f}", va="center", ha="right", fontsize=6.3)
    ax[0].set_yticks(y, [r[1] for r in rows])
    ax[0].set_xlim(15, 85)
    ax[0].set_xlabel("Combined accuracy (%)")
    ax[0].scatter([], [], s=22, color=C["dark"], label="text-sufficient")
    ax[0].scatter([], [], s=22, color="white", edgecolor=C["dark"], lw=1.2, label="image-dependent")
    ax[0].set_ylim(-1.3, len(rows) - 0.5)
    ax[0].legend(loc="lower center", ncol=2, frameon=False, handletextpad=0.3, borderaxespad=0,
                 columnspacing=1.2)
    ax[0].set_title("(a) Accuracy by decision type", loc="left")

    # (b) stronger text encoder: reading improves, looking does not (3 seeds, same four variants)
    common = ["CCEC", "CCEC-G", "Fusion baseline", "Fusion + corr. learning"]
    for col, lab, c, mk in [("acc_text_sufficient", "text-sufficient", C["text"], "o"),
                            ("acc_image_dependent", "image-dependent", C["grey"], "s")]:
        s_vals = 100 * ds.set_index("model").loc[common, col]
        x_vals = 100 * dx.set_index("model").loc[common, col]
        for m in common:
            ax[1].plot([0, 1], [s_vals[m], x_vals[m]], color=c, lw=0.7, alpha=0.45)
        ax[1].plot([0, 1], [s_vals.mean(), x_vals.mean()], color=c, lw=2, marker=mk, ms=4.5, label=lab)
        ax[1].text(1.06, x_vals.mean(), f"{x_vals.mean():.1f}", va="center", fontsize=6.3, color=c)
        ax[1].text(-0.06, s_vals.mean(), f"{s_vals.mean():.1f}", va="center", ha="right", fontsize=6.3, color=c)
    ax[1].set_xticks([0, 1], ["SigLIP-2\ntext", "XLM-R\ntext"])
    ax[1].set_xlim(-0.35, 1.35)
    ax[1].set_ylim(28, 80)
    ax[1].set_ylabel("Combined accuracy (%)")
    ax[1].legend(loc="center", frameon=False, bbox_to_anchor=(0.5, 0.55), handlelength=1.5)
    ax[1].set_title("(b) Stronger text encoder", loc="left")

    # (c) change rate vs change precision (two failure modes)
    if multi:
        mk = {"zero-shot direct": "D", "zero-shot CoT": "D", "few-shot": "^", "LoRA": "v"}
        pts = [(d42, m, None, C["vlm"], mk[m.split(" | ")[1]]) for m in vlm.model]
    else:
        pts = [(d42, "VLM zero-shot (direct)", "VLM, direct", C["vlm"], "D"),
               (d42, "VLM zero-shot (CoT)", "VLM, CoT", C["vlm"], "D")]
    pts += [(ds, m, None, C["grey"], "s") for m in common] + [(dx, m, None, C["ours"], "o") for m in common]
    for df, key, lab, col, mk in pts:
        r = df[df.model == key].iloc[0]
        ax[2].scatter(100 * r.change_rate, 100 * r.change_precision, s=22, color=col, marker=mk,
                      edgecolor="white", lw=0.4, zorder=3)
        if lab:
            ax[2].annotate(lab, (100 * r.change_rate, 100 * r.change_precision), xytext=(-3, 5),
                           textcoords="offset points", fontsize=6.2, color=col, ha="left")
    ann = 100 * float(a.annotator_rate)
    ymax = 30 if multi else 14
    ax[2].axvline(ann, color=C["dark"], lw=0.8, ls="--")
    ax[2].text(ann + 0.4, ymax - 0.4, f"annotators ({ann:.1f}%)", fontsize=6.2, va="top")
    if multi:
        ax[2].scatter([], [], s=20, color=C["vlm"], marker="D", label="VLM, zero-shot")
        ax[2].scatter([], [], s=20, color=C["vlm"], marker="^", label="VLM, few-shot")
        ax[2].scatter([], [], s=20, color=C["vlm"], marker="v", label="VLM, LoRA")
    ax[2].scatter([], [], s=20, color=C["ours"], label="fine-tuned, XLM-R text")
    ax[2].scatter([], [], s=20, color=C["grey"], marker="s", label="fine-tuned, SigLIP-2 text")
    ax[2].legend(loc="upper right", frameon=False, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2,
                 bbox_to_anchor=(1.03, 0.92), markerscale=0.8, fontsize=6.3) if multi else \
        ax[2].legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    ax[2].set_xlim(0, 26)
    ax[2].set_ylim(0, ymax)
    ax[2].set_xlabel("Change rate (%)")
    ax[2].set_ylabel("Change precision (%)")
    ax[2].set_title("(c) Changing the judgement", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, a.out, "fig_diagnostics")


# ─────────────────────────────── Figure: English vs Hindi ───────────────────────────────
def fig_language(a):
    m = pd.read_csv(a.metrics)
    avg = m.groupby(["key", "lang"]).mf1.mean().unstack()
    rows = [("img_convnext", "ConvNeXt-B (image)", C["image"]), ("sota_align", "ALIGN", C["sota"]),
            ("sota_clip", "CLIP", C["sota"]), ("sota_dmcc", "DMCC", C["sota"]),
            ("sig_full", "CCEC-G, SigLIP-2 text", C["grey"]), ("txt_muril", "MuRIL (text)", C["text"]),
            ("sota_robertamft", "RoBERTaMFT", C["sota"]), ("txt_xlmr", "XLM-R (text)", C["text"]),
            ("full", "CCEC-G (ours)", C["ours"]), ("fusion_baseline", "Fusion (ours)", C["fusion"])]
    fig, ax = plt.subplots(figsize=(3.33, 2.55))
    y = np.arange(len(rows))
    for yi, (k, lab, col) in zip(y, rows):
        en, hi = avg.loc[k, "en"], avg.loc[k, "hi"]
        ax.plot([en, hi], [yi, yi], color="#C8C8C8", lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(en, yi, s=24, marker="o", color="white", edgecolor=col, lw=1.3, zorder=3)
        ax.scatter(hi, yi, s=24, marker="o", color=col, zorder=3)
    ax.set_yticks(y, [r[1] for r in rows])
    for t in ax.get_yticklabels():
        if "ours" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xlabel("Avg.$_5$ macro-F1 (%)")
    ax.set_xlim(40, 68)
    ax.scatter([], [], s=20, color="white", edgecolor=C["dark"], lw=1.2, label="English posts")
    ax.scatter([], [], s=20, color=C["dark"], label="Hindi posts")
    ax.legend(loc="lower right", frameon=False, borderaxespad=0.2)
    ax.grid(axis="x", lw=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, a.out, "fig_language")


# ─────────────────────────────── Figure: spider chart per task ───────────────────────────────
def fig_radar(a):
    m = pd.read_csv(a.metrics)
    per = m[m.lang == "all"].groupby("key")[TASKS].mean()
    rows = [("vlm_cot", "VLM, CoT (zero-shot)", C["vlm"], ":"), ("img_convnext", "ConvNeXt-B (image)", C["image"], "--"),
            ("sota_robertamft", "RoBERTaMFT", C["sota"], "-."), ("txt_xlmr", "XLM-R (text)", C["text"], "-"),
            ("full", "CCEC-G (ours)", C["ours"], "-")]
    ang = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(3.33, 2.75), subplot_kw={"polar": True})
    for k, lab, col, ls in rows:
        v = per.loc[k, TASKS].tolist()
        v += v[:1]
        ax.plot(ang, v, color=col, lw=1.6 if k == "full" else 1.1, ls=ls, label=lab)
        if k == "full":
            ax.fill(ang, v, color=col, alpha=0.12)
    ax.set_theta_offset(np.pi / 2)          # Crisis at the top, tasks clockwise
    ax.set_theta_direction(-1)
    ax.set_xticks(ang[:-1], ["Crisis", "Severity", "Inform.", "Humanit.", "Resource"])
    ax.tick_params(axis="x", pad=4)
    ax.set_ylim(15, 80)
    ax.set_yticks([20, 40, 60, 80], ["20", "40", "60", "80"], fontsize=5.8, color=C["grey"])
    ax.set_rlabel_position(36)              # radius labels between Crisis and Severity
    ax.spines["polar"].set_linewidth(0.5)
    ax.grid(lw=0.35, color="#CCCCCC")
    ax.legend(loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False, handlelength=1.8)
    fig.tight_layout()
    save(fig, a.out, "fig_radar")


# ─────────────────────────────── Figure: dataset, text vs image vs combined ───────────────────────────────
def fig_dataset(a):
    # counts from the label distribution table (N = 6,862)
    D = {
        "Informativeness": {"Informative": (6032, 4971, 5718), "Low Informativeness": (830, 1891, 1144)},
        "Humanitarian": {"Affected Individuals": (2459, 1289, 2112), "Infrastructure & Utilities": (1779, 2250, 1981),
                         "Rescue & Medical": (565, 687, 714), "Not Humanitarian": (2059, 2636, 2055)},
        "Resource need": {"Medical": (664, 239, 516), "Rescue Equipment": (771, 848, 858),
                          "Shelter & Relief": (748, 699, 768), "Other": (679, 800, 855),
                          "No Resource Need": (4000, 4276, 3865)},
        "Severity": {"Critical": (2389, 2017, 2294), "Serious": (1989, 1987, 2063), "Moderate": (2484, 2858, 2505)},
    }
    pal = ["#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#999999"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.55), sharey=True)
    for ax, (task, cls) in zip(axes, D.items()):
        names = list(cls)
        arr = np.array([cls[n] for n in names], dtype=float)
        pct = 100 * arr / arr.sum(axis=0, keepdims=True)
        left = np.zeros(3)
        for i, n in enumerate(names):
            ax.barh([2, 1, 0], pct[i], left=left, color=pal[i % len(pal)], edgecolor="white", lw=0.4,
                    height=0.68, label=n)
            for j in range(3):
                if pct[i, j] >= 9:
                    ax.text(left[j] + pct[i, j] / 2, 2 - j, f"{pct[i, j]:.0f}", ha="center", va="center",
                            fontsize=5.8, color="white" if i in (0, 4) else C["dark"])
            left += pct[i]
        ax.set_title(task, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=5.6,
                  handlelength=0.8, handletextpad=0.3, columnspacing=0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([2, 1, 0], ["Text", "Image", "Combined"])
    fig.tight_layout(w_pad=0.6)
    save(fig, a.out, "fig_dataset")


# ─────────────────────────────── Extra figures (need the run folders) ───────────────────────────────
def _hist(root, v, key="val_avg_macro_f1"):
    import glob, json
    H = [[100 * h[key] for h in json.load(open(f))["history"]] for f in sorted(glob.glob(f"{root}/{v}_s4[234]/results.json"))]
    return np.array(H)


def fig_curves(a):
    """Line chart: validation score per epoch (mean over seeds, band = min..max)."""
    fig, ax = plt.subplots(figsize=(3.33, 2.1))
    for root, v, lab, col, ls in [(a.xlmr_runs, "full", "CCEC-G, XLM-R text", C["ours"], "-"),
                                  (a.xlmr_runs, "fusion_baseline", "Fusion, XLM-R text", C["fusion"], "--"),
                                  (a.siglip_runs, "full", "CCEC-G, SigLIP-2 text", C["grey"], "-"),
                                  (a.siglip_runs, "fusion_baseline", "Fusion, SigLIP-2 text", C["grey"], "--")]:
        H = _hist(root, v)
        ep = np.arange(1, H.shape[1] + 1)
        ax.plot(ep, H.mean(0), color=col, ls=ls, lw=1.4, label=lab, marker="o", ms=2.2)
        ax.fill_between(ep, H.min(0), H.max(0), color=col, alpha=0.13, lw=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1 (%)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, a.out, "fig_curves")


def fig_bars(a):
    """Grouped bar chart: macro-F1 per task with standard deviation over seeds."""
    m = pd.read_csv(a.metrics)
    allp = m[m.lang == "all"]
    rows = [("img_convnext", "ConvNeXt-B (image)", C["image"]), ("txt_xlmr", "XLM-R (text)", C["text"]),
            ("sota_robertamft", "RoBERTaMFT", C["sota"]), ("fusion_baseline", "Fusion (ours)", C["fusion"]),
            ("full", "CCEC-G (ours)", C["ours"])]
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    x = np.arange(len(TASKS))
    w = 0.16
    for i, (k, lab, col) in enumerate(rows):
        g = allp[allp.key == k]
        mu, sd = g[TASKS].mean(), g[TASKS].std()
        ax.bar(x + (i - 2) * w, mu, w * 0.92, yerr=sd, color=col, label=lab, capsize=1.6,
               error_kw={"elinewidth": 0.6, "capthick": 0.6}, edgecolor="white", lw=0.3)
    vlm = allp[allp.key == "vlm_cot"]
    if len(vlm):
        ax.scatter(x, vlm[TASKS].iloc[0], marker="_", s=260, color=C["vlm"], lw=1.6, zorder=4,
                   label="VLM, CoT (zero-shot)")
    ax.set_xticks(x, ["Crisis type", "Severity", "Informativeness", "Humanitarian", "Resource need"])
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(15, 85)
    ax.grid(axis="y", lw=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=6, frameon=False, handlelength=1.2,
              columnspacing=1.0)
    fig.tight_layout()
    save(fig, a.out, "fig_bars")


def fig_ablation(a):
    """Diverging bar chart: difference to CCEC-G per variant, mean +- std of the per-seed differences."""
    import glob, json
    from scipy.stats import ttest_rel

    def per_seed(root, v):
        out = {}
        for f in glob.glob(f"{root}/{v}_s4[234]/results.json"):
            sd = int(f.split("_s")[-1].split("/")[0])
            t = json.load(open(f))["test"]
            out[sd] = (100 * t["avg_combined_macro_f1"], 100 * t["cmc"]["macro_f1"])
        return out
    ref = per_seed(a.xlmr_runs, "full")
    rows = [(a.xlmr_runs, "ccec", "without correlation learning (CCEC)"),
            (a.xlmr_runs, "fusion_baseline", "fusion instead of composition"),
            (a.xlmr_runs, "fusion_full", "fusion + correlation learning"),
            (a.xlmr_runs, "full_vlm", "+ VLM knowledge view"),
            (a.xlmr_runs, "fusion_full_vlm", "fusion + corr. + VLM view"),
            (a.comp_runs, "full_vlm_comp", "+ VLM view, trained composition")]
    rows = [r for r in rows if r[0] and glob.glob(f"{r[0]}/{r[1]}_s4*/results.json")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), sharey=True)
    for j, (ax, title) in enumerate(zip(axes, ["$\\Delta$ Avg.$_5$ (points)", "$\\Delta$ CMC macro-F1 (points)"])):
        y = np.arange(len(rows))[::-1]
        ax.set_xlim(-1.3, 0.9) if j == 0 else ax.set_xlim(-4.0, 3.6)
        for yi, (root, v, lab) in zip(y, rows):
            ps = per_seed(root, v)
            seeds = sorted(set(ps) & set(ref))
            d = np.array([ps[s][j] - ref[s][j] for s in seeds])
            pv = ttest_rel([ps[s][j] for s in seeds], [ref[s][j] for s in seeds]).pvalue if len(seeds) > 1 else 1
            col = C["text"] if d.mean() > 0 else C["grey"]          # blue: better than CCEC-G, grey: worse
            sd = d.std(ddof=1) if len(d) > 1 else 0
            ax.barh(yi, d.mean(), xerr=sd, color=col, height=0.6, error_kw={"elinewidth": 0.6, "capsize": 1.6},
                    alpha=0.9)
            if pv < 0.05:                                            # star just beyond the error bar
                sgn = 1 if d.mean() > 0 else -1
                ax.text(d.mean() + sgn * (sd + 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])), yi - 0.08, "*",
                        ha="center", va="center", fontsize=9)
        ax.axvline(0, color=C["dark"], lw=0.7)
        ax.set_xlabel(title)
        ax.grid(axis="x", lw=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(rows))[::-1], [r[2] for r in rows])
    axes[0].set_title("relative to CCEC-G (XLM-R text); * $p<0.05$, paired over seeds", loc="left", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, a.out, "fig_ablation")


def fig_confusion(a):
    """Heatmap: crisis-type confusion matrix of CCEC-G, summed over seeds, row-normalised."""
    import glob
    P = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in
                   sorted(glob.glob(f"{a.xlmr_runs}/full_s4[234]/report/predictions.csv"))], ignore_index=True)
    short = {"Flood / Water Disaster": "Flood", "Cyclone / Storm / Extreme Weather": "Storm",
             "Earthquake & Geological": "Earthquake", "Fire & Industrial Accident": "Fire",
             "Human Casualties": "Casualties", "Infrastructure Damage": "Infrastr.", "Other": "Other"}
    order = list(short)
    cm = pd.crosstab(P["Crisis_Combined_true"], P["Crisis_Combined_pred"]).reindex(index=order, columns=order, fill_value=0)
    pct = 100 * cm.values / cm.values.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.33, 2.75))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(order)):
        for j in range(len(order)):
            if pct[i, j] >= 1:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if pct[i, j] > 55 else C["dark"])
    ax.set_xticks(range(len(order)), [short[o] for o in order], rotation=40, ha="right")
    ax.set_yticks(range(len(order)), [short[o] for o in order])
    ax.set_xlabel("Predicted crisis type")
    ax.set_ylabel("Annotated crisis type")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("% of row", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, a.out, "fig_confusion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="paper_tables_out/main/all_metrics.csv")
    p.add_argument("--diag42", required=True)
    p.add_argument("--diag_x", required=True)
    p.add_argument("--diag_s", required=True)
    p.add_argument("--annotator_rate", default=0.090, help="share of image-dependent decisions (seed 42)")
    p.add_argument("--out", default="figures")
    p.add_argument("--xlmr_runs", default=None, help="runs_xlmr: training curves, ablation and confusion figures")
    p.add_argument("--siglip_runs", default=None, help="runs_v2: SigLIP-2 training curves")
    p.add_argument("--comp_runs", default=None, help="folder with full_vlm_comp_s4* (default: --xlmr_runs)")
    p.add_argument("--only", nargs="*", default=None,
                   help="draw only these figures, e.g. --only diagnostics (names: diagnostics language radar dataset bars)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup_style()
    if a.only:
        for name in a.only:
            globals()[f"fig_{name}"](a)
        return
    fig_diagnostics(a)
    fig_language(a)
    fig_radar(a)
    fig_dataset(a)
    fig_bars(a)
    if a.xlmr_runs:
        a.comp_runs = a.comp_runs or a.xlmr_runs
        fig_ablation(a)
        fig_confusion(a)
        if a.siglip_runs:
            fig_curves(a)


if __name__ == "__main__":
    main()

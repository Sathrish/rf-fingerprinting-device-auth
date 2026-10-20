"""Make figures and a results table from results/*.json (run after run_experiments.py)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    os.makedirs(FIG, exist_ok=True)
    rs, cr, cd, nz, os_, bl, bx = (load(n) for n in
                                   ("random_split", "cross_run", "cross_distance", "noise", "open_set",
                                    "baseline", "baseline_extended"))
    cnn_c, rf_c = "#2f6fa5", "#c97b63"

    # 1. headline comparison of evaluation protocols, CNN vs random forest
    if rs and cr and bl:
        groups = ["Random split\n(original method)", "Later in the\nsame recordings",
                  "New recording\n(run2)", "Unseen distances\n32-62 ft (run2)"]
        cnn = [rs["random_split_test"]["accuracy"], rs["same_model_later_in_same_files"]["accuracy"],
               cr["accuracy_mean"], cd["overall"]["accuracy"] if cd else np.nan]
        rf = [bl["random_split"]["accuracy"], np.nan, bl["cross_run"]["accuracy"],
              bx["unseen_distances"]["overall"]["accuracy"] if bx else np.nan]
        x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for off, vals, c, lab in ((-0.2, cnn, cnn_c, "CNN on spectrograms"),
                                  (0.2, rf, rf_c, "Random forest on impairment features")):
            bars = ax.bar(x + off, [v * 100 for v in vals], 0.4, color=c, label=lab)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.0f}%",
                            ha="center", fontsize=9)
        ax.axhline(100 / 16, ls="--", c="grey", lw=1)
        ax.text(-0.45, 100 / 16 + 1.5, "chance (6.25%)", ha="left", fontsize=8, color="grey")
        ax.set_xticks(x, groups)
        ax.set_ylabel("Device identification accuracy (%)")
        ax.set_ylim(0, 110)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Accuracy falls sharply once test data comes from a different recording")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "evaluation_protocols.png"), dpi=150)
        plt.close(fig)

    # 2. accuracy vs distance (seen and unseen)
    if cr and cd:
        seen = cr["runs"][0]["per_distance"]
        unseen = cd["per_distance"]
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = sorted(int(k) for k in seen)
        xu = sorted(int(k) for k in unseen)
        ax.plot(xs, [seen[str(k)] * 100 for k in xs], "o-", c=cnn_c, label="CNN, training distances (run2)")
        ax.plot(xu, [unseen[str(k)] * 100 for k in xu], "s--", c=cnn_c, label="CNN, unseen distances (run2)")
        if bx:
            pu = bx["unseen_distances"]["per_distance"]
            ax.plot(xu, [pu[str(k)] * 100 for k in xu], "^--", c=rf_c,
                    label="random forest, unseen distances (run2)")
        ax.axhline(100 / 16, ls=":", c="grey")
        ax.set_xlabel("Transmitter-receiver distance (ft)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.set_title("Accuracy by distance (model trained on 2-26 ft)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "accuracy_vs_distance.png"), dpi=150)
        plt.close(fig)

    # 3. noise robustness
    if nz:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(nz["snr_db"], [v * 100 for v in nz["clean_model"]], "o-", c=cnn_c, label="CNN, clean training")
        ax.plot(nz["snr_db"], [v * 100 for v in nz["augmented_model"]], "o--", c=cnn_c,
                label="CNN, noise-augmented training")
        if bx:
            ax.plot(bx["noise"]["snr_db"], [v * 100 for v in bx["noise"]["clean_model"]], "s-", c=rf_c,
                    label="random forest, clean training")
            ax.plot(bx["noise"]["snr_db"], [v * 100 for v in bx["noise"]["augmented_model"]], "s--", c=rf_c,
                    label="random forest, noise-augmented training")
        ax.axhline(100 / 16, ls=":", c="grey")
        ax.invert_xaxis()
        ax.set_xlabel("Added-noise SNR (dB)")
        ax.set_ylabel("Accuracy on run2 (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.set_title("Noise robustness")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "noise_robustness.png"), dpi=150)
        plt.close(fig)

    # 4. confusion matrix
    if cr:
        cm = np.array(cr["runs"][0]["confusion"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(16), cr["devices"], rotation=90, fontsize=7)
        ax.set_yticks(range(16), cr["devices"], fontsize=7)
        ax.set_xlabel("Predicted device")
        ax.set_ylabel("True device")
        ax.set_title("Cross-run confusion matrix (seed 0)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "confusion_cross_run.png"), dpi=150)
        plt.close(fig)

    # 5. open-set ROC
    sp = os.path.join(RES, "open_set_scores.npz")
    if os_ and os.path.exists(sp):
        z = np.load(sp)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for i, f in enumerate((0, 2)):
            y = np.r_[np.ones(len(z[f"fold{f}_cos_known"])), np.zeros(len(z[f"fold{f}_cos_rogue"]))]
            s = np.r_[z[f"fold{f}_cos_known"], z[f"fold{f}_cos_rogue"]]
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=f"fold {i + 1} (rogue: {', '.join(os_['folds'][i]['rogue_devices'][:2])}...)")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("Rogue devices accepted (false-accept rate)")
        ax.set_ylabel("Enrolled devices accepted (true-accept rate)")
        m = os_["mean_over_folds"]["centroid_cosine"]
        ax.set_title(f"Rejecting unseen devices: AUROC {m['auroc']:.2f}, EER {m['eer'] * 100:.0f}%")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "open_set_roc.png"), dpi=150)
        plt.close(fig)

    # results table (markdown) for the README
    rows = []
    if rs:
        rows.append(("CNN, random 80/20 chunk split (original method)", rs["random_split_test"]["accuracy"]))
        rows.append(("Same CNN, chunks from later in the same recordings",
                     rs["same_model_later_in_same_files"]["accuracy"]))
    if cr:
        rows.append((f"CNN, proper cross-run (mean of 2 seeds, std {cr['accuracy_std'] * 100:.1f})",
                     cr["accuracy_mean"]))
    if cd:
        rows.append(("CNN, unseen distances 32-62 ft", cd["overall"]["accuracy"]))
    if bl:
        rows.append(("Random forest on impairment features, random split", bl["random_split"]["accuracy"]))
        rows.append(("Random forest on impairment features, cross-run", bl["cross_run"]["accuracy"]))
    if bx:
        rows.append(("Random forest on impairment features, unseen distances 32-62 ft",
                     bx["unseen_distances"]["overall"]["accuracy"]))
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Setup | Accuracy |\n|---|---|\n")
        for name, acc in rows:
            f.write(f"| {name} | {acc * 100:.1f}% |\n")
    print(open(os.path.join(RES, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()

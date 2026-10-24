import sys, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
sys.path.insert(0, "python")
import lr_inference as lr

C = {"s1":"#2a78d6","s2":"#eb6834","ink":"#0b0b0b","ink2":"#52514e",
     "muted":"#898781","grid":"#e1e0d9","axis":"#c3c2b7","surface":"#fcfcfb"}
mpl.rcParams.update({"figure.facecolor":C["surface"],"axes.facecolor":C["surface"],
    "savefig.facecolor":C["surface"],"savefig.dpi":200,"figure.dpi":110,
    "font.family":"sans-serif","font.sans-serif":["Segoe UI","DejaVu Sans"]})


# Figure 8: why one shared Q format fails

m = lr.load_model("models/lr_weights.npz")
Q88, Q115 = lr.QFormat(16,8), lr.QFormat(16,1)
q88,_ = lr.quantize(m.coef, Q88)

order = np.argsort(np.abs(m.coef))
names = [m.feature_names[i] for i in order]
mag   = np.abs(m.coef)[order]
dead  = (q88[order] == 0)

fig, ax = plt.subplots(figsize=(8.8, 4.8))
y, xmin = np.arange(len(mag)), 5e-4
t88, t115 = Q88.resolution/2, Q115.resolution/2

ax.axvspan(xmin, t88, color=C["s2"], alpha=0.07, zorder=1)
ax.axvline(t88, ls="--", lw=1.6, color=C["s2"], zorder=2)

for i,(v,d) in enumerate(zip(mag, dead)):
    col = C["s2"] if d else C["s1"]
    ax.plot([xmin, v], [i, i], color=col, lw=2, zorder=3, solid_capstyle="round")
    ax.plot(v, i, "o", ms=9, color=col, zorder=4,
            markeredgecolor=C["surface"], markeredgewidth=2)
    ax.annotate(f"{v:.6f}", xy=(v, i), xytext=(12, 0), textcoords="offset points",
                va="center", fontsize=9, color=C["ink2"])

ax.annotate(f"Q8.8 rounds to zero below {t88:.5f}", xy=(t88, -1.0),
            ha="center", va="center", fontsize=9, color=C["s2"])
ax.text(0.985, 0.04, f"Q1.15 threshold is {t115:.7f}, off-scale to the left",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=C["muted"])
ax.annotate("silently dropped" + "\n" + "from the model", xy=(mag[0], 0.12),
            xytext=(0.42, 0.17), textcoords="axes fraction",
            fontsize=9.5, color=C["s2"],
            arrowprops=dict(arrowstyle="->", color=C["s2"], lw=1.4,
                            connectionstyle="arc3,rad=-0.25"))

ax.set_xscale("log"); ax.set_xlim(xmin, 1.4); ax.set_ylim(-1.6, len(mag)-0.4)
ax.set_yticks(y, names, fontsize=10)
ax.set_xlabel("|weight|  (log scale)")
ax.set_title("One shared Q8.8 format deletes a feature from the model",
             fontsize=12, color=C["ink"], pad=12)
ax.xaxis.grid(True, color=C["grid"], lw=0.8); ax.set_axisbelow(True)
for s_ in ("top","right","left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(C["axis"]); ax.tick_params(colors=C["ink2"], length=0)
fig.tight_layout(); fig.savefig("figures/fig8_quantization.png")
print("saved figures/fig8_quantization.png")


# Figure 9: consensus against centralization


FLOOR = 0.4124
rows = [("Relay R2 alone", 0.4827, "relay"), ("Relay R3 alone", 0.4845, "relay"),
        ("Relay R1 alone", 0.4921, "relay"), ("Relay R4 alone", 0.4945, "relay"),
        ("OR consensus, 4 relays", 0.5214, "or"),
        ("Centralized, all relays", 0.5724, "ref")]
role = {"relay":C["s1"], "or":C["s2"], "ref":C["muted"]}

labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
cols   = [role[r[2]] for r in rows]
y = np.arange(len(rows))

fig, ax = plt.subplots(figsize=(8.8, 4.4))
ax.axvspan(0, FLOOR, color=C["muted"], alpha=0.10, zorder=1)
ax.barh(y, vals, height=0.6, color=cols, zorder=3)
ax.axvline(FLOOR, ls="--", lw=1.6, color=C["ink2"], zorder=4)

for i,(v,c) in enumerate(zip(vals, cols)):
    ax.annotate(f"{v:.4f}", xy=(v, i), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=C["ink2"])

ax.annotate("4 bits on the wire", xy=(0.012, 4), fontsize=9.5,
            color="#ffffff", va="center", ha="left", zorder=5)
ax.annotate("32 features + 4 flags", xy=(0.012, 5), fontsize=9.5,
            color="#ffffff", va="center", ha="left", zorder=5)
ax.annotate(f"always-predict-Attack floor  {FLOOR:.4f}", xy=(FLOOR, -0.95),
            xytext=(-8, 0), textcoords="offset points",
            ha="right", va="center", fontsize=9, color=C["ink2"])

ax.set_xlim(0, 0.70); ax.set_ylim(-1.5, len(rows)-0.35)
ax.set_yticks(y, labels, fontsize=10)
ax.set_xlabel("macro F1 on the held-out test set (data6, data11, data15)")
ax.set_title("Four relays voting recover most of a centralized model, using one bit each",
             fontsize=12, color=C["ink"], pad=12)
ax.xaxis.grid(True, color=C["grid"], lw=0.8); ax.set_axisbelow(True)
for s_ in ("top","right","left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(C["axis"]); ax.tick_params(colors=C["ink2"], length=0)
fig.tight_layout(); fig.savefig("figures/fig9_consensus.png")
print("saved figures/fig9_consensus.png")

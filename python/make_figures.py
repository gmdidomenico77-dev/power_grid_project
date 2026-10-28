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

# ------------------------------------------------------------------------
# Figure 10: system architecture
# ------------------------------------------------------------------------

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(13.2, 6.2))
ax.set_xlim(0, 104); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, label, sub=None, ec=C["ink2"], lw=1.4, fs=10.5, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=C["surface"], ec=ec, lw=lw, zorder=3))
    tc = tc or C["ink"]
    if sub:
        ax.text(x+w/2, y+h*0.62, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
        ax.text(x+w/2, y+h*0.26, sub, ha="center", va="center", fontsize=8.6, color=C["ink2"], zorder=4)
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs, color=tc, zorder=4)

def arrow(x1, y1, x2, y2, label=None, color=None, lw=1.6, fs=8.4):
    color = color or C["ink2"]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+2.6, label, ha="center", va="bottom",
                fontsize=fs, color=C["ink2"], zorder=4)

ROWS = [78, 58, 38, 18]
BUS = 69.0

ax.add_patch(Rectangle((20, 9), 17, 82, fc=C["muted"], alpha=0.07, ec="none", zorder=1))
ax.add_patch(Rectangle((41, 9), 47, 82, fc=C["s1"], alpha=0.06, ec=C["s1"],
                       lw=1.2, ls=(0,(5,3)), zorder=1))

ax.text(28.5, 94.0, "ARM processing system", ha="center", fontsize=9.8, color=C["ink"])
ax.text(28.5, 90.2, "software", ha="center", fontsize=8.4, color=C["muted"])
ax.text(64.5, 94.0, "Zynq-7020 programmable logic", ha="center", fontsize=9.8, color=C["s1"])
ax.text(64.5, 90.2, "460 ns deterministic  ·  36 DSP  ·  922 LUT (1.7%)  ·  0 BRAM",
        ha="center", fontsize=8.4, color=C["ink2"])

for i, y in enumerate(ROWS):
    box(2, y-5, 14, 10, f"Relay R{i+1}", "PMU @ 120 Hz", ec=C["axis"])
    arrow(16.2, y, 21.8, y, "9 features" if i == 0 else None)
    box(22, y-5, 13, 10, "cap · scale", "quantize", ec=C["muted"])
    arrow(35.2, y, 43.8, y, "9 x int16" if i == 0 else None)
    box(44, y-5, 20, 10, "LR core", "9 MAC, sign test", ec=C["s1"])
    ax.plot([64.2, BUS], [y, y], color=C["s2"], lw=2.0, zorder=2, solid_capstyle="round")
    if i == 0:
        ax.text(66.6, y+2.6, "1 bit", ha="center", va="bottom", fontsize=8.4, color=C["ink2"])

ax.plot([BUS, BUS], [ROWS[-1], ROWS[0]], color=C["s2"], lw=2.0, zorder=2, solid_capstyle="round")
ax.plot(BUS, 50, "o", ms=6, color=C["s2"], zorder=4)
ax.text(BUS-0.8, 50, "4 bits  ", ha="right", va="center", fontsize=8.4, color=C["ink2"])

arrow(BUS, 50, 75.8, 50, color=C["s2"], lw=2.0)
box(76, 41, 9, 18, "OR", None, ec=C["s2"], lw=2.0, fs=13, tc=C["s2"])
ax.text(80.5, 37.6, "any relay", ha="center", fontsize=8.2, color=C["ink2"])

arrow(85.2, 50, 94, 50, color=C["s2"], lw=2.2)
ax.text(95.5, 52.2, "1 = Attack",  ha="left", va="center", fontsize=10, color=C["s2"])
ax.text(95.5, 47.4, "0 = Natural", ha="left", va="center", fontsize=10, color=C["ink2"])

ax.text(52, 3.0, "Each relay reduces 9 local measurements to 1 bit inside the fabric. "
                 "Four bits cross the network; one gate decides.",
        ha="center", fontsize=10.2, color=C["ink"])

fig.tight_layout()
fig.savefig("figures/fig10_architecture.png", bbox_inches="tight")
fig.savefig("figures/fig10_architecture.svg", bbox_inches="tight")
print("saved figures/fig10_architecture.png + .svg")

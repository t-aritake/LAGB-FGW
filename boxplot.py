import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

df = pd.read_csv('./results/results_artificial.csv')

# fig, ax = plt.subplots(1, 1, figsize=(6, 2))
# palette = sns.color_palette(['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'])
# sns.set_palette(palette)
# flierprops = {
#     'marker': 'o',
#     'markerfacecolor': 'white',
#     'markeredgewidth': 0.5,
#     'markeredgecolor': 'k',
#     'markersize': 2}
# sns.boxplot(data=df, x='datasets', y='accuracy', hue='methods', linewidth=0.5,
#             flierprops=flierprops)
# 
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# ax.legend(fontsize=6, ncols=4)
# ax.set_ylim(0, 1.1)
method_order = [
    "DSFT", "DSTN", "DAEVS", "Naive", "OT", "GW",
    "FGWOT", "LA-FGW", "GB-FGW", "LAGB-FGW (ours)"
]
dataset_order = ["Blobs", "Two circles", "Two moons", "Two spirals"]

g = sns.catplot(
    data=df,
    kind="box",
    x="methods",          # 各手法
    y="accuracy",
    hue="methods",
    col="datasets",       # ← datasetごとにsubplot
    order=method_order,
    hue_order=method_order,
    col_order=dataset_order,
    col_wrap=2,
    sharey=True,
    height=3.0,
    aspect=1.35,
    legend=True
)

g.set_titles("")

# x軸ラベル消す（冗長なので）
g.set_axis_labels("", "accuracy")
g.set_xticklabels(rotation=45)
# g.set_ylim(0, 1.1)

# ---------- captionを追加 ----------
labels = ["(a) Blobs", "(b) Two circles", "(c) Two moons", "(d) Two spirals"]

for ax, label in zip(g.axes.flatten(), labels):
    bbox = ax.get_position()   # figure 座標でのsubplot位置
    x = (bbox.x0 + bbox.x1) / 2
    y = bbox.y0 - 0.015         # 下に少し余白

    g.fig.text(
        x, y,
        label,
        ha="center",
        va="top",
        fontsize=12
    )

# legendを右に1個だけ
g._legend.set_bbox_to_anchor((1.01, 0.5))
g._legend.set_title("")

# plt.show()
plt.savefig('./boxplot_artificial.pdf')
plt.close('all')

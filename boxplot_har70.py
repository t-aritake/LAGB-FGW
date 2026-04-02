import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy.stats
from scipy.stats import wilcoxon
import pdb


def wilcoxon_by_participant(df, m1, m2):
    out = {}
    for p, g in df.groupby("participants"):
        wide = g.pivot(index="trials", columns="methods", values="accuracy")
        wide = wide[[m1, m2]].dropna()
        if len(wide) < 2:
            out[p] = numpy.nan
        else:
            out[p] = wilcoxon(wide[m1], wide[m2]).pvalue
    return out


def p_to_sig(p):
    if p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'


df = pd.read_csv('./results/results_har70.csv')

p_daevs = wilcoxon_by_participant(df, "DAEVS", "LAGB-FGW")
p_fgwgd = wilcoxon_by_participant(df, "GB-FGW", "LAGB-FGW")

method_order = ["DAEVS", "Naive", "OT", "GW", "FGWOT", "LA-FGW", "GB-FGW", "LAGB-FGW"]

fig, ax = plt.subplots(1, 1, figsize=(6, 2))
palette = sns.color_palette("colorblind", 9)
# palette = sns.color_palette(['#E69F00', '#56B4E9', '#009E73', '#F0E442',
#                             '#0072B2', '#D55E00', '#CC79A7', '#000000'])
sns.set_palette(palette)
flierprops = {
    'marker': 'o',
    'markerfacecolor': 'white',
    'markeredgewidth': 0.5,
    'markeredgecolor': 'k',
    'markersize': 2}
# sns.boxplot(data=df, x='participants', y='accuracy', hue='methods', linewidth=0.5,
#             flierprops=flierprops)


# # 3. 有意差アノテーション
# for i in range(res_list[0].shape[0]):
#     x1_offset = -0.35   # 箱の横幅の半分くらい
#     x2_offset = 0.35
#     y_offset = 0.02  # 線・* をデータ上限からどれだけ上にずらすか
#     # DAEVS vs LA-FGW-GD
#     if p_daevs[i] < 0.05:
#         # x1, x2 は参加者 i を中心に左右にずらす
#         x1, x2 = i + x1_offset, i + x2_offset
#         # y はその参加者の全メソッドの最大 accuracy + オフセット
#         y = df[df['participants'] == i]['accuracy'].max() + y_offset
#         # 線を描画
#         ax.plot([x1, x1, x2, x2],
#                 [y, y + 0.005, y + 0.005, y],
#                 lw=0.5, c='k')
#
#         symbol = p_to_sig(p_daevs[i])
#         ax.text((x1 + x2) / 2, y + 0.005, f'{symbol}',
#                 ha='center', va='bottom', fontsize=6)
#
#     # FGW-GD vs LA-FGW-GD
#     x1_offset = 0.23
#     if p_fgwgd[i] < 0.05:
#         x1, x2 = i + x1_offset, i + x2_offset
#         y = df[df['participants'] == i]['accuracy'].max() + y_offset * 2
#
#         symbol = p_fgwgd[i]
#         ax.plot([x1, x1, x2, x2],
#                 [y, y + 0.005, y + 0.005, y],
#                 lw=0.5, c='k')
#         symbol = p_to_sig(p_fgwgd[i])
#         ax.text((x1 + x2) / 2, y + 0.005, f'{symbol}',
#                 ha='center', va='bottom', fontsize=6)
#
#
# handles, labels = ax.get_legend_handles_labels()

# 並び替えの準備
# n = len(labels)
# n_rows = 2
# n_cols = int(numpy.ceil(n / n_rows))

# row-major → column-major インデックス変換
# column_major_indices = []
# for c in range(n_cols):
#     for r in range(n_rows):
#         idx = r * n_cols + c
#         if idx < n:
#             column_major_indices.append(idx)
#
# # 並び替えたハンドルとラベルで凡例を再描画
# handles_ordered = [handles[i] for i in column_major_indices]
# labels_ordered = [labels[i] for i in column_major_indices]
#
# ax.legend(handles_ordered, labels_ordered, fontsize=4, ncol=n_cols,
#           loc='lower left')  # 位置調整はお好みで
#
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# # ax.legend(fontsize=4, ncols=4)
# ax.set_ylim(0, 1.2)
# plt.savefig(f'./boxplots/boxplot_har70.pdf', bbox_inches='tight', pad_inches=0)
# plt.close('all')

df["methods"] = pd.Categorical(
    df["methods"],
    categories=method_order,
    ordered=True,
)

participants = sorted(df["participants"].unique())

# ---- facet plot (4列) ----
g = sns.catplot(
    data=df,
    kind="box",
    x="methods",
    hue="methods",
    y="accuracy",
    col="participants",
    col_wrap=2,
    order=method_order,
    col_order=participants,
    sharey=True,
    # linewidth=0.5, fliersize=2,
    height=2.2, aspect=1.1,
    legend=True
)

for ax in g.axes.flat:
    ax.set_xlabel("")                      # x軸ラベル（methods など）を消す
    ax.tick_params(axis="x", labelbottom=False)

g.set_titles("")  # 上の "participants = 0" を消す（captionを下に付けるなら不要）
g.set_axis_labels("", "accuracy")

# caption例：下中央に "participant i" を置く
# for ax, p in zip(g.axes.flat, g.col_names):
#     ax.text(0.5, -0.25, f"(participant {p})", transform=ax.transAxes,
#             ha="center", va="top", fontsize=8)
#     ax.tick_params(axis="x", labelrotation=45, labelsize=6)
#     ax.tick_params(axis="y", labelsize=7)
for ax, label in zip(g.axes.flatten(), g.col_names):
    bbox = ax.get_position()   # figure 座標でのsubplot位置
    x = (bbox.x0 + bbox.x1) / 2
    y = bbox.y0 - 0.01         # 下に少し余白

    g.fig.text(
        x, y,
        f"participants {label+1}",
        ha="center",
        va="top",
        fontsize=10
    )

# --- 有意差（各subplot=participant） ---
for ax, p in zip(g.axes.flat, g.col_names):
    p = int(p)  # col_namesが文字列なら
    ymax = df[df["participants"] == p]["accuracy"].max()
    y0 = ymax + 0.02

    # DAEVS vs LA-FGW-GD
    s = p_to_sig(p_daevs.get(p, numpy.nan))
    if s:
        x1 = method_order.index("DAEVS")
        x2 = method_order.index("LAGB-FGW")
        ax.plot([x1, x1, x2, x2], [y0, y0 + 0.01, y0 + 0.01, y0], lw=0.6, c="k")
        ax.text((x1 + x2) / 2, y0 + 0.01, s, ha="center", va="bottom", fontsize=8)

    # FGW-GD vs LA-FGW-GD（少し上に）
    s = p_to_sig(p_fgwgd.get(p, numpy.nan))
    if s:
        y1 = y0 + 0.05
        x1 = method_order.index("GB-FGW")
        x2 = method_order.index("LAGB-FGW")
        ax.plot([x1, x1, x2, x2], [y1, y1 + 0.01, y1 + 0.01, y1], lw=0.6, c="k")
        ax.text((x1 + x2) / 2, y1 + 0.01, s, ha="center", va="bottom", fontsize=8)
# # ---------- captionを追加 ----------
# labels = ["(a) id=1", "(b) id=2", "(c) id=3", "(d) id=4", "(e) id=5", "(f) id=6", "(g) id=7", "(h) id=8"]
#
# for ax, label in zip(g.axes.flatten(), labels):
#     bbox = ax.get_position()   # figure 座標でのsubplot位置
#     x = (bbox.x0 + bbox.x1) / 2
#     y = bbox.y0 - 0.015         # 下に少し余白
#
#     g.fig.text(
#         x, y,
#         label,
#         ha="center",
#         va="top",
#         fontsize=10
#     )

# legend 1個だけ
handles, labels = g.axes.flat[0].get_legend_handles_labels()
# n = len(participants)
# ncols, nrows = 4, int(numpy.ceil(n / 4))
#
# fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows), sharey=True)
#
# axes = numpy.array(axes).reshape(-1)
#
# palette = sns.color_palette("colorblind", len(method_order))
# flierprops = dict(marker='o', markerfacecolor='white', markeredgewidth=0.5,
#                   markeredgecolor='k', markersize=2)
#
# for k, pid in enumerate(participants):
#     ax = axes[k]
#     sub = df[df["participants"] == pid].copy()
#
#     # そのparticipantだけ横に methods で箱ひげ
#     sns.boxplot(
#         data=sub, x="methods", y="accuracy",
#         order=method_order, ax=ax,
#         linewidth=0.5, flierprops=flierprops, palette=palette)
#
#     wide = sub.pivot_table(index="trials", columns="methods", values="accuracy")
#
#     # 有意差描画 helper
#     def annotate_sig(m1, m2, y_add=0.02):
#         if m1 not in wide.columns or m2 not in wide.columns:
#             return
#         pair = wide[[m1, m2]].dropna()
#         if len(pair) < 2:
#             return
#         p = wilcoxon(pair[m1], pair[m2]).pvalue
#         sig = p_to_sig(p)
#         if not sig:
#             return
#         x1 = method_order.index(m1)
#         x2 = method_order.index(m2)
#         y = sub["accuracy"].max() + y_add
#         ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=0.6, c="k")
#         ax.text((x1 + x2) / 2, y + 0.002, sig, ha="center", va="bottom", fontsize=7)
#
#     annotate_sig("DAEVS", "LAGB-FGW", y_add=0.01)
#     annotate_sig("GB-FGW", "LAGB-FGW", y_add=0.03)   # FGWGD相当がどれかに合わせて修正
#
# # 余ったaxes消す
# for j in range(len(participants), len(axes)):
#     fig.delaxes(axes[j])
#
# # legendは1個だけ
# handles, labels = axes[0].get_legend_handles_labels()
# # seaborn boxplotはlegend持たないことが多いので、自前で作るならpatch拾うより簡単にやる:
# # → ここでは「legendなし」で運用するのが普通。必要なら catplot に寄せる。
#
# fig.subplots_adjust(right=0.98, bottom=0.08, hspace=0.35, wspace=0.25)
plt.savefig("boxplot_har70_by_participant.pdf", bbox_inches='tight')
plt.close()
#
# #     stats= (
# #     df.groupby(["participants", "methods"])["accuracy"]
# #     .agg(
# #         median="median",
# #         q25=lambda x: x.quantile(0.25),
# #         q75=lambda x: x.quantile(0.75),
# #     )
# #     .reset_index()
# #     .sort_values(["participants", "methods"])   # now follows method_names order
# # )
# #
# #     # round / format
# #     for col in ["median", "q25", "q75"]:
# #     stats[col]= stats[col].map(lambda v: f"{v:.3f}")
# #
# #     # --- emit LaTeX ------------------------------------------------
# #     header = r"""\begin{table}[t]
# # \centering
# # \footnotesize
# # \caption{Per-dataset accuracy (median and inter-quartile range).}
# # \label{tab:toy_iqr}
# # \renewcommand{\arraystretch}{0.9}
# # \begin{tabular}{l l c c c}
# # \toprule
# # \textbf{Dataset} & \textbf{Method} & \textbf{Median} & $Q_{25}$ & $Q_{75}$\\
# # \midrule
# # """
# #
# #     rows = "\n".join(
# # f"{row.participants} & {row.methods} & {row.median} & {row.q25} & {row.q75} \\\\"
# # for row in stats.itertuples(index=False)
# # )
# #
# #     footer = r"""\bottomrule
# # \end{tabular}
# # \end{table}"""
# #
# #     latex_table = header + rows + "\n" + footer
# #     print(latex_table)

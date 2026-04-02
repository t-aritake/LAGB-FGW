import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('./results/runs_har70_beta.csv')
df['dataset'] = df['har70_artificial_shift'].apply(eval).map(lambda x: x['target_id'] - 500)

# df2 = pd.read_csv('./results/runs(5).csv')

# df.loc[df['dataset'] == 'Two circles', 'Accuracy'] = df2.iloc[:160]['Accuracy'].values
target = "beta"

stats = (
    df.groupby(["dataset", target])["Accuracy"]
    .agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    .reset_index()
)

plt.figure(figsize=(6, 3))

sns.set_style("whitegrid")
datasets = stats["dataset"].unique()
palette = sns.color_palette("colorblind", len(datasets))
markers = ('o', '^', 'v', '>', '<', 'x', 's', '+')

for i, (color, ds) in enumerate(zip(palette, datasets)):
    sub = stats[stats["dataset"] == ds].sort_values(target)

    # median line
    plt.plot(
        sub[target],
        sub["median"],
        marker=markers[i],
        markersize=3,
        lw=0.5,
        label=ds,
        color=color
    )

    # IQR band
    plt.fill_between(
        sub[target],
        sub["q25"],
        sub["q75"],
        alpha=0.25,
        color=color
    )

# plt.xscale("log")   # ← α は log が自然
plt.ylim(0.7, 1.05)

if target == "alpha":
    plt.xlabel(r"$\alpha\ (\beta=0.1,\ K=4)$", fontsize=16)
elif target == "beta":
    plt.xlabel(r"$\beta\ (\alpha=0.01,\ K=4)$", fontsize=16)
elif target == "n_neighbors":
    plt.xlabel(r"$K\ (\alpha=0.01, \beta=0.1)$", fontsize=16)
plt.ylabel("accuracy", fontsize=16)

plt.legend(frameon=True, ncol=4)
plt.tight_layout()
plt.savefig(f"appendix_{target}_har70.pdf", bbox_inches='tight')

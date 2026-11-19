import numpy as np
from scipy import stats


def design(table):
    ids = table.animal.fillna("").astype(str).str.strip()
    if (ids == "").any():
        return "unknown", "动物编号不完整：无法确认独立性，仅描述，不计算 p 值。"
    membership = table.assign(animal=ids).groupby("animal").group.nunique()
    if membership.max() == 1:
        return "independent", "不同组动物编号不重叠：先在同一动物内等权平均细胞，再比较独立动物。"
    if membership.min() == table.group.nunique():
        return "paired", "按相同动物编号配对；请确认是同一动物而不是不同组重复使用了编号。"
    return "partial", "动物跨组且配对不完整：建议核对编号并采用含动物随机效应的混合模型；此处仅描述。"


def metric_figure(project, metric):
    from .plots import _group_bar, _label, new_figure
    table = project.cell_table().dropna(subset=[metric])
    fig = new_figure()
    if table.empty:
        return fig, "没有该指标的有效值。"
    table = table.assign(animal=table.animal.fillna("").astype(str).str.strip())
    kind, reason = design(table)
    units = table if kind == "unknown" else table.groupby(["animal", "group"], as_index=False)[metric].mean()
    groups = list(table.group.unique())
    samples = [units.loc[units.group == group, metric].to_numpy() for group in groups]
    estimate = len(groups) == 2 and kind in ("independent", "paired") and min(map(len, samples)) >= 2
    if estimate:
        axis, difference_axis = fig.subplots(1, 2, width_ratios=[2, 1])
    else:
        axis = fig.subplots()
    rng = np.random.default_rng(0)
    for index, values in enumerate(samples):
        _group_bar(axis, index, values, rng)
    labels = [f"{group}\n(n={len(values)})" for group, values in zip(groups, samples)]
    axis.set(xticks=range(len(groups)), xticklabels=labels, ylabel=_label(metric), title="Mean ± SEM")
    note = reason + "\n每点：" + ("细胞" if kind == "unknown" else "动物") + "；" + "，".join(f"{g}: n={len(v)}" for g,v in zip(groups,samples))
    if estimate:
        first, second = samples
        difference = second.mean() - first.mean()
        if kind == "paired":
            paired = units.pivot(index="animal", columns="group", values=metric)
            changes = paired[groups[1]] - paired[groups[0]]
            se = changes.sem()
            df = len(changes) - 1
        else:
            a, b = (np.var(v, ddof=1) / len(v) for v in samples)
            se = np.sqrt(a + b)
            denominator = a**2 / (len(first) - 1) + b**2 / (len(second) - 1)
            df = (a + b)**2 / denominator if denominator else np.inf
        margin = stats.t.ppf(.975, df) * se
        comparison = f"{groups[1]} − {groups[0]}"
        difference_axis.axhline(0, color="0.65", linestyle="--", linewidth=.7)
        difference_axis.errorbar(0, difference, yerr=margin, fmt="o", color="black", capsize=3)
        difference_axis.set(xticks=[0], xticklabels=[comparison], ylabel=_label(metric), title="Mean difference (95% CI)")
        method = "配对 t 区间" if kind == "paired" else "Welch t 区间"
        note += f"\n均值差（{comparison}）：{difference:.4g}，95% CI [{difference-margin:.4g}, {difference+margin:.4g}]（{method}）。"
    if kind not in ("independent", "paired") or len(groups)<2 or min(map(len,samples))<2:
        return fig, note + "\n未进行显著性检验。"
    if kind == "paired":
        pivot = units.pivot(index="animal", columns="group", values=metric)[groups]
        samples = [pivot[group].to_numpy() for group in groups]
        test, name = (stats.wilcoxon, "Wilcoxon 配对符号秩") if len(groups)==2 else (stats.friedmanchisquare, "Friedman 重复测量")
    else:
        test, name = (stats.mannwhitneyu, "Mann–Whitney U") if len(groups)==2 else (stats.kruskal, "Kruskal–Wallis")
    try:
        probability = test(*samples).pvalue
        note += f"\n推荐并执行：{name}，p={probability:.4g}" if np.isfinite(probability) else "\n数据无变异，无法给出有效 p 值。"
    except ValueError:
        note += "\n数据不足或无变异，未报告 p 值。"
    return fig, note

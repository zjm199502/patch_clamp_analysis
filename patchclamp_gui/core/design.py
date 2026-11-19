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
    from .plots import new_figure
    table = project.cell_table().dropna(subset=[metric])
    fig = new_figure()
    axis = fig.subplots()
    if table.empty:
        return fig, "没有该指标的有效值。"
    kind, reason = design(table)
    units = table if kind == "unknown" else table.groupby(["animal", "group"], as_index=False)[metric].mean()
    groups = list(units.group.unique())
    samples = [units.loc[units.group == group, metric].to_numpy() for group in groups]
    for index, values in enumerate(samples):
        axis.scatter(np.full(len(values), index), values, alpha=.7)
        axis.plot([index-.18, index+.18], [values.mean()]*2, color="black")
    axis.set(xticks=range(len(groups)), xticklabels=groups, ylabel=metric)
    note = reason + "\n每点：" + ("细胞" if kind == "unknown" else "动物") + "；" + "，".join(f"{g}: n={len(v)}" for g,v in zip(groups,samples))
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

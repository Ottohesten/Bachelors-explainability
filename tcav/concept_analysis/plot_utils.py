from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def assemble_scores(tcav_scores, experimental_sets, idx, score_layer, score_type,):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(tcav_scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])
    score_list = np.array(score_list)
    return score_list


def plot_tcav_scores(all_experimental_sets, all_tcav_scores, concept_names=None, title="", score_type="sign_count", alpha=0.05, only_significant=True,
                     with_error=True, plt_name="plot", file_type="svg"):
    tcav_score0 = next(iter(all_tcav_scores[0].values()))
    layers = list(tcav_score0.keys())
    scores_mean = {layer: [] for layer in layers}
    scores_std = {layer: [] for layer in layers}
    stars = {layer: [] for layer in layers}
    concept_short_names = []
    for experimental_sets, tcav_scores in zip(all_experimental_sets, all_tcav_scores):
        concept_short_codes = [int(str(concept.id)[:5]) for concept in experimental_sets[0]]
        concept_short_names_next = [concept.name[:-4] for concept in experimental_sets[0]]
        concept_short_names.append(concept_short_names_next[0])

        for layer in layers:
            pos_scores = assemble_scores(tcav_scores, experimental_sets, 0, layer, score_type)
            neg_scores = assemble_scores(tcav_scores, experimental_sets, 1, layer, score_type)
            # t-test
            _, pval = ttest_ind(pos_scores, neg_scores)

            # mann whitney u rank test
            _, pval_whit = mannwhitneyu(pos_scores, neg_scores)

            print(f"pval: {pval}, pval_whit: {pval_whit}")
            # Bonferroni correction
            m = 2
            alpha = alpha / m
            # Get non-significant and significant scores
            # not_significant = np.mean(pos_scores) < np.mean(neg_scores) or pval >= alpha
            not_significant = pval >= alpha
            stars[layer].append(~not_significant)
            if not_significant and only_significant:
                scores_mean[layer].append(0.0)
                scores_std[layer].append(0.0)
            else:
                scores_mean[layer].append(np.mean(pos_scores))
                scores_std[layer].append(np.std(pos_scores))

    if not concept_names:
        scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_short_names)
        scores_std_df = pd.DataFrame(data=scores_std, index=concept_short_names)
    else:
        scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_names)
        scores_std_df = pd.DataFrame(data=scores_std, index=concept_names)

    fig = plt.figure(figsize=(18, 6))
    # plt.rcParams.update({'font.size': 22})
    ax = plt.subplot()
    if with_error:
        scores_mean_df.plot.bar(ax=ax, rot=0, yerr=scores_std_df)
    else:
        scores_mean_df.plot.bar(ax=ax, rot=0)

    containers = []
    for container in ax.containers:
        if isinstance(container, BarContainer):
            containers.append(container)
    for container, star in zip(containers, stars.values()):
        labels = []
        for s_idx, s in enumerate(star):
            if s:
                labels.append("*")
                current_color = container.patches[s_idx].get_facecolor()
                new_color = current_color
                # new_color = [c for c in current_color[:-1]]
                # new_color.append(0.25)
                container.patches[s_idx].set_facecolor(new_color)
            else:
                labels.append("")
        ax.bar_label(container, labels=labels, size=18, color=container.patches[0].get_facecolor())
    ax.set_ylim(bottom=0.0, top=1.0)

    ax.set_ylabel("TCAV score", fontsize=18)
    ax.legend(title="Network layer")
    ax.legend(loc="upper right", title="Network layer")
    ax.set_title(title, fontsize=32)
    # make x_ticks bigger
    ax.tick_params(axis='x', labelsize=18)

    plt.tight_layout()
    plt.savefig(f"figures/{plt_name}.{file_type}")
    plt.show()
    plt.close()

    return scores_mean_df, scores_std_df


# using plotly instead of matplotlib
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu

def plot_tcav_scores_plotly(all_experimental_sets, all_tcav_scores, concept_names=None, title="", score_type="sign_count", alpha=0.05, only_significant=True,
                     with_error=True, plt_name="plot", file_type="svg"):
    tcav_score0 = next(iter(all_tcav_scores[0].values()))
    layers = list(tcav_score0.keys())
    scores_mean = {layer: [] for layer in layers}
    scores_std = {layer: [] for layer in layers}
    stars = {layer: [] for layer in layers}
    concept_short_names = []
    for experimental_sets, tcav_scores in zip(all_experimental_sets, all_tcav_scores):
        concept_short_codes = [int(str(concept.id)[:5]) for concept in experimental_sets[0]]
        concept_short_names_next = [concept.name[:-4] for concept in experimental_sets[0]]
        concept_short_names.append(concept_short_names_next[0])

        for layer in layers:
            pos_scores = assemble_scores(tcav_scores, experimental_sets, 0, layer, score_type)
            neg_scores = assemble_scores(tcav_scores, experimental_sets, 1, layer, score_type)
            # t-test
            _, pval = ttest_ind(pos_scores, neg_scores)
            # mann whitney u rank test
            _, pval_whit = mannwhitneyu(pos_scores, neg_scores)

            print(f"pval: {pval}, pval_whit: {pval_whit}")
            pval = pval_whit
            # print(pval)
            # Bonferroni correction
            m = 2
            alpha = alpha / m
            # Get non-significant and significant scores
            # not_significant = np.mean(pos_scores) < np.mean(neg_scores) or pval >= alpha
            not_significant = pval >= alpha
            stars[layer].append(~not_significant)
            if not_significant and only_significant:
                scores_mean[layer].append(0.0)
                scores_std[layer].append(0.0)
            else:
                scores_mean[layer].append(np.mean(pos_scores))
                scores_std[layer].append(np.std(pos_scores))

    if not concept_names:
        scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_short_names)
        scores_std_df = pd.DataFrame(data=scores_std, index=concept_short_names)
    else:
        scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_names)
        scores_std_df = pd.DataFrame(data=scores_std, index=concept_names)

    stars = {layer: ["*" if star else "" for star in stars[layer]] for layer in layers}

    fig = go.Figure()
    for layer in layers:
        fig.add_trace(go.Bar(
            x=concept_short_names,
            y=scores_mean[layer],
            # error_y=dict(type='data', array=scores_std[layer]),
            name=layer,
            text=stars[layer],
            # textposition='outside',
            # text color
        ))

    fig.update_layout(
        width=1200,
        height=720,
        # barmode='group',
        title=title,
        yaxis_title='TCAV score',
        xaxis_title='Concept',
        font=dict(
            size=18,
        ),

    )
    # print(stars)
    # change x tick labels
    fig.update_xaxes(tickvals=[i for i in range(len(concept_short_names))])
    fig.update_xaxes(ticktext=concept_names)

    # set y-axis to always go from 0 to 1
    fig.update_yaxes(range=[0, 1])
    # change tick angle
    # fig.update_xaxes(tickangle=0)
    # save figure
    fig.write_image(f"figures/{plt_name}_plotly.{file_type}")
    fig.show()

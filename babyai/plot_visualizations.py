import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import defaultdict

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", show_cbar=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# mp = '/HDD/nhashemi/neural_production_systems/babyai/visualizations/BabyAI-GoToLocal-v0_ppo_NPS_vanilla_gru_mem_seed42_22-08-25-11-38-28'
# mp = '/HDD/nhashemi/neural_production_systems/babyai/visualizations/BabyAI-GoToLocal-v0_ppo_NPS_vanilla_gru_mem_seed42_22-08-25-11-41-23'
# mp = '/HDD/nhashemi/neural_production_systems/babyai/visualizations/BabyAI-GoToLocal-v0_ppo_NPS_vanilla_VQ_mem_seed1_22-08-27-13-41-48'
# num_rules = 8

# mp = '/HDD/nhashemi/neural_production_systems/babyai/visualizations/BabyAI-OpenTwoDoors-v0_ppo_NPS_vanilla_gru_mem_seed42_22-08-26-13-27-06'
# mp = '/HDD/nhashemi/neural_production_systems/babyai/visualizations/BabyAI-OpenTwoDoors-v0_ppo_NPS_vanilla_gru_mem_seed42_22-08-26-13-27-50'
# mp = '/HDD/nhashemi/neural_production_systems/babyai/visualizations/BabyAI-OpenTwoDoors-v0_ppo_NPS_vanilla_gru_mem_seed1_22-08-25-11-51-38'
# num_rules = 4


mp = '/HDD/nhashemi/NPS/neural_production_systems/babyai/visualizations/BabyAI-GoToLocal-v0_ppo_NPS_vanilla_gru_mem_seed1_22-10-09-20-21-31'
num_rules = 21
# mp = '/HDD/nhashemi/NPS/neural_production_systems/babyai/visualizations/BabyAI-GoToLocal-v0_ppo_NPS_vanilla_gru_mem_seed1_22-10-09-20-14-56'
# num_rules = 8

ep_res = pickle.load(open(f'{mp}_episode_results.pkl', 'rb'))
d_results = {}
for logs in ep_res:
    # if logs['return_per_episode'][-1] > 0:
    if logs['mission'] in d_results:
        d_results[logs['mission']].append(logs)
    else:
        d_results[logs['mission']] = [logs]

def plot_rule_embs():
    instr2rule = {}
    for m in d_results:
        instr2rule[m] = d_results[m][0]['rule_embs_per_episode'][0][0][0].cpu().numpy()
    rule_embs = np.concatenate(list(instr2rule.values()), axis=0)

    color2colormap = {
        'red': 'red',
        'green': 'green',
        'grey': 'gray',
        'blue': 'royalblue',
        'yellow': 'goldenrod',
        'purple': 'purple'
    }
    shape2stylemap = {
        'ball': 'o',
        'box': 'X',
        'key': '^'
    }

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rule_embs.reshape(-1, 64))
    fig = plt.figure(dpi=900)
    ax = fig.add_subplot(111)

    ax.scatter(pca_result[:, 0], pca_result[:, 1], 
        c=[np.mod(x, num_rules) for x in range(len(pca_result))], cmap='summer', s=3, alpha=0.8)
    for i in range(num_rules):
        if num_rules == 8 and (i == 7 or i == 2):
            xr = -2
            yr = -2
        else:
            xr = yr = 0
        plt.annotate(f'{(np.mod(i, num_rules) + 1)}', xy=(pca_result[i, 0], pca_result[i, 1]), 
            xytext=(pca_result[i, 0]+xr, pca_result[i, 1]+yr), fontsize=15) 

    plt.show()
    plt.savefig(f'{mp}_embs.png')
    fig = plt.figure(figsize=(18, 7), dpi=700)
    for r in range(num_rules):
        pr = pca_result.reshape(-1, num_rules, 2)[:, r, :]
        ax = fig.add_subplot(240+r+1)
        if num_rules == 4:
            ax_other = fig.add_subplot(240+r+5)
        for i in range(len(pr)):
            mission = list(instr2rule.keys())[i].split(' ')
            if num_rules == 8:
                color = mission[-2]
                shape = mission[-1]
                print(mission, color2colormap[color], shape2stylemap[shape])
                ax.scatter(pr[i, 0], pr[i, 1], 
                color=color2colormap[color], marker=shape2stylemap[shape], s=80, 
                alpha=0.8, facecolors='none' if 'a' in mission else color2colormap[color])
                ax.set_title(f'Rule #{r+1}', fontsize=15)
            elif num_rules == 4:
                color1 = color2colormap[mission[2]]
                color2 = color2colormap[mission[-2]]
                ax.scatter(pr[i, 0], pr[i, 1], 
                    color=color1, s=80, alpha=0.8)
                ax_other.scatter(pr[i, 0], pr[i, 1], 
                    color=color2, s=80, alpha=0.8)
                ax.set_title(f'Rule #{r+1}, first color', fontsize=15)
                ax_other.set_title(f'Rule #{r+1}, second color', fontsize=15)

    fig.tight_layout()
    plt.show()
    plt.savefig(f'{mp}_embs_rules.png')

def correlate_rule_ids(comp_step=0):
    """
    ys, x in xs:
    (5, x): key
    (6, x): ball
    (7, x): box

    xs, y in ys:
    (y, 0): red
    (y, 1): green
    (y, 2): blue
    (y, 3): purple
    (y, 4): yellow
    (y, 5): grey

    (0, 0): unseen
    (2, 5): obstacle
    (1, 0): empty tile
    """
    code2name = {
        '0-0': 'unseen',
        '5-2': 'obstacle',
        '0-1': 'empty',
    } 
    for p in [(4, 'door'), (5, 'key'), (6, 'ball'), (7, 'box')]:
        for q in [(0, 'red'), (1, 'green'), (2, 'blue'), (3, 'purple'), (4, 'yellow'), (5, 'grey')]:
            code2name[f'{q[0]}-{p[0]}'] = f'{q[1]} {p[1]}'


    d_results = defaultdict(lambda: [1e-10]*num_rules)
    d_results_targetted = defaultdict(lambda: [1e-10]*num_rules)
    d_results_untargetted = defaultdict(lambda: [1e-10]*num_rules)
    # for ent in ['0-0', '5-2', '0-1']:
    #     d_results_targetted[ent] = torch.tensor([0]*num_rules)
    for logs in ep_res:
        for step in range(len(logs['observations_per_episode'][0])):
        # if logs['return_per_episode'][-1] > 0:
            # step = -1
            obs = logs['observations_per_episode'][0][step]
            rule_ids = logs['rule_ids_per_episode'][0][step][comp_step]
            entities = obs['image'].reshape(-1, 3)
            for i in range(len(entities)):
                e = entities[i]
                r = rule_ids[i]
                ent = f'{e[1]}-{e[0]}'
                d_results[ent][r] += 1
                if code2name[ent] in logs['mission']:
                    d_results_targetted[ent][r] += 1
                else:
                    d_results_untargetted[ent][r] += 1
    d_results = dict(sorted(d_results.items()))
    d_results_targetted = dict(sorted(d_results_targetted.items()))
    d_results_untargetted = dict(sorted(d_results_untargetted.items()))

    for d in [d_results, d_results_targetted, d_results_untargetted]:
        for k in d:
            if sum(d[k]) > 0:
                d[k] = torch.tensor([v / sum(d[k]) for v in d[k]])

    fig, ax = plt.subplots(1, 3, figsize=(10,7))
    x = torch.stack(list(d_results.values()), dim=0).numpy()
    # im, cbar = heatmap(x, 
    #                 [code2name[k] for k in d_results.keys()], 
    #                 list(range(num_rules)), ax=ax[0],
    #                 cmap="YlGn", show_cbar=False)
    sns.clustermap(pd.DataFrame(x.T, columns=[code2name[k] for k in d_results.keys()]), metric="euclidean")
    ax[0].set_title('Both')
    x = torch.stack(list(d_results_targetted.values()), dim=0).numpy()
    # im, cbar = heatmap(x, 
    #                 [' '] * 21, 
    #                 list(range(num_rules)), ax=ax[1],
    #                 cmap="YlGn", show_cbar=False)
    sns.clustermap(pd.DataFrame(x.T, columns=[code2name[k] for k in d_results_targetted.keys()]), metric="euclidean")
    ax[1].set_title('Target entity')
    x = torch.stack(list(d_results_untargetted.values()), dim=0).numpy()
    # im, cbar = heatmap(x, 
    #                 [' '] * 21, 
    #                 list(range(num_rules)), ax=ax[2],
    #                 cmap="YlGn", cbarlabel=f"Rule prob. per entity (cstep={comp_step+1})", show_cbar=False)
    sns.clustermap(pd.DataFrame(x.T, columns=[code2name[k] for k in d_results_untargetted.keys()]), metric="euclidean")
    ax[2].set_title('Non-target entity')
    # texts = annotate_heatmap(im)
    fig.tight_layout()
    plt.show()
    plt.savefig(f'{mp}_rule_entity_heatmap_target_vs_nontarget_{comp_step+1}.png')

    return d_results

correlate_rule_ids(comp_step=0)
# plot_rule_embs()

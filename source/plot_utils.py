import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_json_for_pareto(results_json):
    """
    Read a single json file and collect probe and classifier accuracy for each value
    of beta
    :param results_json:
    :return: numpy array (t, probe_accuracy, classifier_accuracy)
    """
    results_list = []

    with open(results_json) as json_file:
        results = json.load(json_file)

        for t, res_probe_t in results['classifier'].items():
            classifier_accuracy = res_probe_t["0"]['validation']['199']['accuracy']
            probes_accuracy = results['probes'][str(t)]["0"]['accuracy']

            results_list.append(np.array([float(t), probes_accuracy, classifier_accuracy]))

        summary = np.vstack(results_list)

        return summary


def plot_pareto_front(results_json, outfolder, tag=None):
    """
    Plot probe and classifier accuracy for each value of beta
    :param results_json:
    :param outfolder:
    :param tag:
    :return:
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = ['o', '<', 's', '>']

    for i, result in enumerate(results_json):
        name = result[0]
        summary = read_json_for_pareto(result[1])

        axe = ax.scatter(summary[:, 1], summary[:, 2], c=summary[:, 0], cmap=plt.get_cmap('cividis'),
                         label=name, marker=markers[i], s=100)
        if name in ['AFBC4', 'AFBC3']:
            ax.plot(summary[:, 1], summary[:, 2], color='#af7ac5', label='Pareto Front')

    plt.xlabel('Accuracy of Auditor', fontsize=20)
    plt.ylabel('Accuracy of Classifier', fontsize=20)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(axe, cax=cax, orientation='vertical', pad=0.1)
    cax.tick_params(labelsize=5)
    cax.set_label(r'$\beta$')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax.legend(loc='lower right', prop={'size': 14})

    plt.savefig(f'{outfolder}/pareto_front_{tag}.png')
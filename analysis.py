import os
import argparse

from source.plot_utils import plot_pareto_front

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', required=True)

    args = parser.parse_args()

    outfolder = f'figures/tests'
    os.makedirs(outfolder, exist_ok=True)

    if args.figure == 'mnist-rot-pareto':
        results_list = [('AFBC', 'summaries/predict_outcome_sensitive/mnist_rot_comp_all5/20210501005411_predict_outcome_sensitive_0_89469.json'),
                        ('AFBC2', 'summaries/predict_outcome_sensitive/mnist_rot_comp_all5/20210501120317_predict_outcome_sensitive_0_44609.json'),
                        ('AFBC3', 'summaries/predict_outcome_sensitive/mnist_rot_comp_all5/20210503161136_predict_outcome_sensitive_0_85662.json'),
                        ('AFBC4', 'summaries/predict_outcome_sensitive/mnist_rot_comp_all5/20210503101523_predict_outcome_sensitive_0_72134.json')]
        plot_pareto_front(results_list, outfolder, tag='mnist_rot')
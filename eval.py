import argparse
import datetime
import json
import os

import yaml
import numpy as np

from experiments.probe import Probe
from source.utils import get_logger, disable_warnings


logger = get_logger(__name__)
disable_warnings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--fairness', type=float)
    parser.add_argument('--tmax', type=float)
    parser.add_argument('--random_order', type=bool)

    args = parser.parse_args()
    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    with open(args.config_path, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    seed = 0
    if args.seed is not None:
        seed = args.seed

    beta = None
    if args.beta is not None:
        beta = args.beta

    fairness = None
    if args.fairness is not None:
        fairness = args.fairness

    experiment_type = config_dict['experiment_type']
    name = config_dict['name']
    logging_dir = config_dict['logging_dir']

    checkpoint_dir = None
    if 'checkpoints' in config_dict:
        checkpoint_dir = config_dict['checkpoints']
        checkpoint_dir = checkpoint_dir + '/' + name
        os.makedirs(checkpoint_dir, exist_ok=True)

    rind = np.random.randint(0, 10**5)

    outfolder = f'{logging_dir}/summaries/{experiment_type}/{name}'
    os.makedirs(outfolder, exist_ok=True)
    outfile = f'{outfolder}/{tstamp}_{experiment_type}_{seed}_{rind}.json'

    PR = Probe(config_dict['autoencoder'], config_dict['probes_list'],
               beta=beta, decode=config_dict['decode'],
               seed=seed, fairness=fairness, checkpoints=checkpoint_dir,
               task_list=config_dict['task_list'])

    threshold_range = np.linspace(0, args.tmax, 10, dtype=float)
    for t in threshold_range:

        PR.threshold = t

        PR.train_rep_loader = PR.generate_representation(PR.train_dset, shuffle=True)
        PR.test_rep_loader = PR.generate_representation(PR.test_dset, shuffle=True)
        PR.validate_rep_loader = PR.generate_representation(PR.validate_dset, shuffle=True)

        PR.probe_sensitive()
        PR.classify_from_representation()

    writer = PR.results
    json_results = json.dumps(writer)

    f = open(outfile, "w")
    f.write(json_results)
    f.close()
    logger.info(f'Save results and parameters in {outfile}')


import yaml
import copy

from torch.utils.data.dataloader import DataLoader
from sklearn.manifold import TSNE

from source.dataset import *
from source.model import Model
from source.auditors import *
from source.losses import *
from source.utils import get_logger

logger = get_logger(__name__)

class Probe(object):

    def __init__(self, autoencoder_path, probes_list,
                 decode=False, beta=None, seed=None, fairness=None, checkpoints=None, task_list=None):

        with open(autoencoder_path, 'r') as stream:
            self.config_autoencoder = yaml.load(stream, Loader=yaml.SafeLoader)

        if beta is not None:
            self.config_autoencoder['beta'] = beta

        if fairness is not None:
            self.config_autoencoder['gamma'] = fairness

        if seed is not None:
            self.config_autoencoder['data']['seed'] = seed

        save = False
        if checkpoints is not None:
            save = True

        self.results = {}
        self.results['autoencoder'] = copy.deepcopy(self.config_autoencoder['net'])
        self.results['loss'] = copy.deepcopy(self.config_autoencoder['loss'])
        self.results['method'] = self.config_autoencoder['experiment']
        self.results['beta'] = self.config_autoencoder['beta']

        self.results['training'] = {}
        self.results['validation'] = {}
        self.results['training']['rec_loss'] = {}
        self.results['validation']['rec_loss'] = {}
        self.results['validation']['bit_rate'] = {}

        self.autoencoder = Model.from_dict(self.config_autoencoder)
        n_epochs = self.config_autoencoder['n_epochs']

        if 'run' in self.config_autoencoder:
            run =  self.config_autoencoder['run']
            logger.info(f'Loading checkpoint {run}')
            checkpoint = torch.load(run, map_location='cpu')
            self.autoencoder.net.load_state_dict(checkpoint['model_state_dict'])

        self.transfer = False
        if 'transfer' in self.config_autoencoder:
            self.transfer = self.config_autoencoder['transfer']
            self.config_autoencoder['data']['transfer'] = True

        self.transfer_small = False
        if 'transfer_small' in self.config_autoencoder:
            self.transfer_small = self.config_autoencoder['transfer_small']
            self.config_autoencoder['data']['transfer_small'] = True

        if 'prun' in self.config_autoencoder:
            prun =  self.config_autoencoder['prun']
            logger.info(f'Loading checkpoint {prun}')
            checkpoint = torch.load(prun, map_location='cpu')
            self.autoencoder.pmodel.load_state_dict(checkpoint['model_state_dict'])

        if 'code' in self.config_autoencoder:
            code = self.config_autoencoder['code']
            logger.info(f'Loading symbols {code}')
            self.autoencoder.net.code = torch.from_numpy(np.load(code)).float()

        self.probes_list = probes_list

        if self.transfer | self.transfer_small | (task_list is not None):
            self.task_list = task_list
        else:
            self.taks_list = probes_list

        self.decode = decode

        dataname = self.config_autoencoder['data'].pop('name')
        self.results['dataname'] = dataname

        train_dset = globals()[dataname].from_dict(self.config_autoencoder['data'], type='train')
        test_dset = globals()[dataname].from_dict(self.config_autoencoder['data'], type='test')
        validate_dset = globals()[dataname].from_dict(self.config_autoencoder['data'], type='validate')

        train_loader = DataLoader(train_dset,
                                  batch_size=self.config_autoencoder['batch_size'],
                                  shuffle=True)
        validate_loader = DataLoader(validate_dset,
                                 batch_size=self.config_autoencoder['batch_size'])

        logger.info(f'Train autoencoder to generate representations')
        self.autoencoder.train(train_loader, validate_loader, n_epochs, self.results,
                                   save=save, chkpt_dir=checkpoints)
        self.autoencoder.net.eval()

        rec_loss, mask_loss, accuracy, bitrate = self.autoencoder.eval(validate_loader)
        self.results['validation']['bit_rate'] = bitrate.mean().item()
        self.results['validation']['rec_loss_final'] = rec_loss[0].item()

        self.device = self.autoencoder.device
        self.nclass = self.config_autoencoder['nclass_outcome']
        self.nclass_sensitive = self.config_autoencoder['nclass_sensitive']

        self.train_dset = train_dset
        self.test_dset = test_dset
        self.validate_dset = validate_dset
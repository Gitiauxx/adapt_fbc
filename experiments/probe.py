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
                                  shuffle=True,
                                  num_workers=8)
        validate_loader = DataLoader(validate_dset,
                                 batch_size=self.config_autoencoder['batch_size'])

        logger.info(f'Train autoencoder to generate representations')
        self.autoencoder.train(train_loader, validate_loader, n_epochs, self.results,
                                   save=save, chkpt_dir=checkpoints)
        self.autoencoder.net.eval()

        self.results['validation']['bit_rate'] = {}
        self.results['validation']['rec_loss_final'] = {}

        for beta in [0.0, self.autoencoder.beta]:
            val_loss, accuracy, s_loss, entr_loss, active_bits = self.autoencoder.eval(validate_loader, beta)
            self.results['validation']['bit_rate'][beta] = entr_loss.item()
            self.results['validation']['rec_loss_final'][beta] = val_loss.item()

        self.device = self.autoencoder.device
        self.nclass = self.config_autoencoder['nclass_outcome']
        self.nclass_sensitive = self.config_autoencoder['nclass_sensitive']

        self.train_dset = train_dset
        self.test_dset = test_dset
        self.validate_dset = validate_dset

        self.threshold = 0
        self.validate_rep_loader = None
        self.test_rep_loader = None

    def generate_representation(self, dset, shuffle=True, order=True):
        """
        Using the representation mapping from the autoencoder,
        generate representation z or (if self.decode=True), decoder(z)
        :return:
        """
        rep_generator = RepDataset(dset, self.autoencoder, device=self.device, threshold=self.threshold)
        rep_loader = DataLoader(rep_generator, shuffle=shuffle,
                                batch_size=self.config_autoencoder['batch_size'])

        self.zdim = rep_generator.zdim
        return rep_loader

    def probe_sensitive(self):
        """
        for each path in probe_list construct the corresponding probe
        and train it to predict sensitive attribute
        :return:
        """
        if 'probes' not in self.results:
            self.results['probes'] = {}

        t = self.threshold
        self.results['probes'][str(t)] = {}

        for i, config_probe_path in enumerate(self.probes_list):
            self.results['probes'][str(t)][i] = {}

            with open(config_probe_path, 'r') as stream:
                config_probe = yaml.load(stream, Loader=yaml.SafeLoader)

            self.results['probes'][str(t)][i]["depth"] = config_probe["classifier"]["depth"]
            self.results['probes'][str(t)][i]["width"] = config_probe["classifier"]["width"]

            logger.info(f'Probing with model {config_probe["classifier"]["name"]} of '
                            f'width {config_probe["classifier"]["width"]} and '
                            f'depth {config_probe["classifier"]["depth"]}')

            n_epochs = config_probe['n_epochs']
            config_probe['classifier']['zdim'] = self.config_autoencoder['net']['zk'] * self.config_autoencoder['net']['k']

            config_probe['classifier']['nclass'] = self.nclass_sensitive

            probe = ProbeFairness.from_dict(config_probe)
            probe.train(self.validate_rep_loader, self.test_rep_loader, n_epochs,
                            nclass=self.nclass_sensitive, writer=self.results['probes'][str(t)][i])

            probe_loss, accuracy = probe.eval(self.test_rep_loader, nclass=self.nclass_sensitive)
            self.results['probes'][str(t)][i]['accuracy'] = accuracy.item()
            self.results['probes'][str(t)][i]['loss'] = probe_loss.item()

    def classify_from_representation(self, transfer=False):
        """
        for each path in probe_list construct the corresponding classifier
        and train it to predict an outcome as defined in the corresponding
        dataset
        :return:
        """
        if 'classifier' not in self.results:
            self.results['classifier'] = {}

        t = self.threshold
        self.results['classifier'][str(t)] = {}

        for i, config_classifier_path in enumerate(self.task_list):
            self.results['classifier'][str(t)][i] = {}

            with open(config_classifier_path, 'r') as stream:
                config_classifier = yaml.load(stream, Loader=yaml.SafeLoader)

            self.results['classifier'][str(t)][i]["depth"] = config_classifier["classifier"]["depth"]
            self.results['classifier'][str(t)][i]["with"] = config_classifier["classifier"]["width"]

            logger.info(f'Probing with model {config_classifier["classifier"]["name"]} of '
                            f'width {config_classifier["classifier"]["width"]} and '
                            f'depth {config_classifier["classifier"]["depth"]}')

            n_epochs = config_classifier['n_epochs']

            config_classifier['classifier']['zdim'] = self.config_autoencoder['net']['zk'] * self.config_autoencoder['net']['k']
            config_classifier['device'] = self.autoencoder.device
            config_classifier['classifier']['nclass'] = self.nclass

            probe = ProbePareto.from_dict(config_classifier)
            probe.train(self.validate_rep_loader, self.test_rep_loader, n_epochs,
                            writer=self.results['classifier'][str(t)][i], nclass=self.nclass)
            accuracy, _ = probe.eval(self.test_rep_loader, nclass=self.nclass)
            self.results['classifier'][str(t)][i]['accuracy'] = accuracy.item()


class ProbeFairness(object):
    """
    Generate a downstream probe/user/test function train to predict sensitive
    attribute from representation
    """

    def __init__(self, classifier, loss, learning_rate=0.01, device='cpu'):

        device = torch.device(device)
        self.classifier = classifier.to(device)
        self.loss = loss
        self.device = device

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate, betas=(0.5, 0.999),
                                              weight_decay=1e-5)

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a model input configuration from a eval config dictionary

        Parameters
        ----------
        config_dict : configuration dictionary

        """
        name_net = config_dict['classifier'].pop('name')
        classifier = globals()[name_net].from_dict(config_dict['classifier'])

        name_loss = config_dict['loss'].pop('name')
        loss = globals()[name_loss].from_dict(config_dict['loss'])

        learning_rate = config_dict['learning_rate']
        device = config_dict['device']

        return cls(classifier, loss, learning_rate=learning_rate, device=device)

    def optimize_parameters(self, x, sensitive, **kwargs):
        """
        implement one forward pass and one backward propagation pass
        Parameters
        ----------
        x: (B, zdim)
        target (B, 1)

        Returns
        -------
        """
        prelogits = self.classifier.forward(x)
        loss = self.loss.forward(sensitive, prelogits)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_loader, validation_loader, n_epochs, nclass=1, writer=None):

        writer['validation'] = {}

        for epoch in range(n_epochs):
            train_loss = 0

            for _, batch in enumerate(train_loader):
                input = batch['input_mean'].to(self.device)
                sensitive = batch['sensitive'].to(self.device)

                if nclass > 1:
                    sensitive = torch.argmax(sensitive, -1)

                loss = self.optimize_parameters(input, sensitive)
                train_loss += loss.detach().cpu() * len(input) / len(train_loader.dataset)

            logger.info(f'Epoch: {epoch} Train loss: {train_loss}')

            if (epoch % 5 == 0) | (epoch == n_epochs - 1):
                val_loss, accuracy = self.eval(validation_loader, nclass=nclass)
                logger.info(f'Epoch: {epoch} Probe Accuracy: {accuracy}')
                logger.info(f'Epoch: {epoch} Validation loss: {val_loss}')

                writer['validation'][epoch] = {}
                writer['validation'][epoch]['accuracy'] = accuracy.item()

    def eval(self, data_loader, nclass=1):
        """
        Measure reconstruction loss for self.net and accuracy and demographic parity for self.classifier
        :param data_loader: torch.DataLoader
        :return: probe loss, accuracy, demographic_parity
        """

        probe_loss = 0
        accuracy = 0

        for _, batch in enumerate(data_loader):
            input = batch['input_mean'].to(self.device)
            sensitive = batch['sensitive'].to(self.device)

            s = sensitive
            if nclass > 1:
                s = torch.argmax(sensitive, -1)

            output = self.classifier.forward(input)

            loss = self.loss.forward(s, output)
            probe_loss += loss.detach().cpu() * len(input) / len(data_loader.dataset)

            if nclass <= 1:
                pred = (output >= 0.5).float()
            else:
                pred = output.argmax(-1)

            acc = (pred == s).float().mean(0)
            accuracy += acc * len(input) / len(data_loader.dataset)

        return probe_loss, accuracy

class ProbePareto(ProbeFairness):
    """
    Generate a downstream user with a classification task.
    from a sample of the representation distribution.
    It is used to generate Pareto-front, i.e. accuracy/demographic parity trade-off
    """

    def train(self, train_loader, validation_loader, n_epochs, writer=None, nclass=1):

        writer['validation'] = {}

        for epoch in range(n_epochs):
            train_loss = 0

            for _, batch in enumerate(train_loader):
                x = batch['input'].to(self.device)
                input = batch['input_mean'].to(self.device)

                outcome = batch['target'].to(self.device)

                loss = self.optimize_parameters(input, outcome)
                train_loss += loss.detach().cpu() * x.shape[0] / len(train_loader.dataset)

            logger.info(f'Epoch: {epoch} Train loss: {train_loss}')

            if (epoch % 5 == 0) | (epoch == n_epochs - 1):
                val_loss, accuracy = self.eval(validation_loader, nclass=nclass)
                logger.info(f'Epoch: {epoch} Classifier Accuracy: {accuracy}')
                writer['validation'][epoch] = {}
                writer['validation'][epoch]['accuracy'] = accuracy.item()

    def eval(self, data_loader, nclass=1):
        """
        Measure reconstruction loss for self.net and accuracy and demographic parity for self.classifier
        :param data_loader: torch.DataLoader
        :return: probe loss, accuracy, demographic_parity
        """

        probe_loss = 0
        accuracy = 0

        for _, batch in enumerate(data_loader):
            input = batch['input_mean'].to(self.device)
            y = batch['target'].to(self.device)

            output = self.classifier.forward(input)

            loss = self.loss.forward(y, output)
            probe_loss += loss.detach().cpu() * len(input) / len(data_loader.dataset)

            if nclass <= 1:
                pred = (output >= 0.5).float()
            else:
                pred = output.argmax(-1)

            acc = (pred == y).float().mean(0)
            accuracy += acc * len(input) / len(data_loader.dataset)

        return probe_loss, accuracy

"""
This file contains the code needed to train a classifier on top of the BERTweet model.

It enables hyperparameter optimization (using --mode=train) as well as training 
and subsequent testing (--mode=test)
"""

# !/usr/bin/env python3

# standard library imports
from absl import app, flags, logging
import warnings

# 3rd party imports
import sh
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
import nlp
import transformers
from datasets import ClassLabel, Value

# silence warnings
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# define flags
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 400, '')
flags.DEFINE_string('model', 'vinai/bertweet-base', '')
flags.DEFINE_string('train_ds', 'df_all_train', '')
flags.DEFINE_string('val_ds', 'df_all_val', '')
flags.DEFINE_string('test_ds', 'df_all_test', '')
flags.DEFINE_string('mode', 'default_mode', '')
flags.DEFINE_integer('n_trials', 0, '')

FLAGS = flags.FLAGS

# define model class

class UserClassifier(pl.LightningModule):
    def __init__(self,
                 lr=None,
                 momentum=None,
                 batch_size=None,
                 seq_length=None):

        super().__init__()
        
        # import pretrained BERTweet model
        self.pretrained_model = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=FLAGS.model)

        # freeze the BERTweet model weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # define the classification head
        self.classifier = th.nn.Sequential(th.nn.Linear(self.pretrained_model.pooler.dense.out_features, 2))

        self.save_hyperparameters()

        # define the loss
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

        # define the hyperparams (can be set in main function)
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.seq_length = seq_length


    def prepare_data(self):
        """
        used to prepare the datasets (tokenize descriptions)
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model)
        #tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

        def _tokenize(x):
            """
            Takes a row of a dataset and creates column 'input_ids', by tokenizing the column 'description'.

            :param x: a single row from the dataset
            :return: The row of the dataset with additional column 'input_ids'
            """
            # tokenize the profile description
            x['input_ids'] = tokenizer.encode(
                x['description'],
                max_length=self.seq_length,
                padding='max_length',
                truncation=True
            )

            return x

        def _prepare_ds(dataset_name):
            """
            Defines the datatypes of the features and tokenizes the whole dataset

            :param dataset_name: A string containing the dataset name.
            :return: the tokenized dataset which can be used for training
            """

            path = f'data/{dataset_name}.csv'

            ds = nlp.load_dataset('csv',
                                  data_files=path,
                                  features=nlp.Features({'description': Value('string'),
                                                         'is_gen_pub': ClassLabel(num_classes=2),
                                                         'source': Value('string')}
                                                        ),
                                  split='train[:100%]'
                                  )

            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'is_gen_pub'])
            return ds

        self.train_ds, self.val_ds, self.test_ds = map(_prepare_ds, (FLAGS.train_ds, FLAGS.val_ds, FLAGS.test_ds))

    def forward(self, input_ids):
        """
        The forward function used in training. Takes inputs and produces model outputs.

        :param input_ids: The tokenized input descriptions
        :return: Logits, which are interpreted by the loss function as class probabilities.
        """

        # create a dummy tensor filled with 0
        token_type_ids = th.zeros_like(input_ids) 

        # create a mask, masking the padding tokens 
        mask = (input_ids != 1).float()

        # create the features obtained from the BERTweet model
        with th.no_grad():
            features = self.pretrained_model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)

        # pass features through classification head to obtain outputs
        logits = self.classifier(features.pooler_output)

        return logits

    def training_step(self, batch, batch_idx):
        """
        The training step. Takes a batch of outputs and labels. Computes loss and logs it.

        :param batch: a batch of tokenized inputs
        :return: A dictionary containing the loss and some logs.
        """

        # compute predictions
        logits = self.forward(batch['input_ids'])

        # take labels
        labels = batch['is_gen_pub'].long()

        # compute loss
        loss = self.loss(logits, labels).mean()

        self.logger.experiment.add_scalar("Train/Loss",
                                          loss,
                                          self.current_epoch)

        return {'loss': loss, 'log': {'train_loss': loss}}



    def validation_step(self, batch, batch_idx):
        """
        The validation step. Takes a batch of outputs and labels. Computes loss and logs it.

        :param batch: a batch of tokenized inputs
        :return: A dictionary containing the loss and some logs.
        """
        
        # compute predictions
        logits = self.forward(batch['input_ids'])

        # take labels
        labels = batch['is_gen_pub'].long()
        
        # compute loss
        loss = self.loss(logits, labels)

        # compute accuracy
        acc = logits.argmax(-1) == labels
        acc = acc.float()

        # log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return {'loss': loss, 'acc': acc}


    def test_step(self, batch, batch_idx):
        """
        The test step. Takes a batch of outputs and labels. Computes loss and logs it.

        :param batch: a batch of tokenized inputs
        :return: A dictionary containing performance measures
        """
        
        # compute predictions
        logits = self.forward(batch['input_ids'])

        # take labels
        labels = batch['is_gen_pub'].long()
        
        # compute loss
        loss = self.loss(logits, labels)

        # compute accuracy
        acc = logits.argmax(-1) == labels
        acc = acc.float()
        
        # calculate true positives
        tp = (logits.argmax(-1) == labels) & (labels == 1)
        tp = tp.float()
        
        # calculate true negatives
        tn = (logits.argmax(-1) == labels) & (labels == 0)
        tn = tn.float()

        # calculate false positives
        fp = (logits.argmax(-1) != labels) & (labels == 0)
        fp = fp.float()
        
        # calculate false negatives
        fn = (logits.argmax(-1) != labels) & (labels == 1)
        fn = fn.float()

        return {'loss': loss,
                'acc': acc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn}


    def validation_epoch_end(self, outputs):
        """
        Called at the end of a validation epoch. Accumulates epoch performance.

        :param outputs: model predictions
        :return: A dictionary containing the loss and some logs
        """
        
        # accumulate loss
        loss = th.cat([o['loss'] for o in outputs], 0).mean()

        # accumulate accuracy
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        
        # put them together
        out = {'val_loss': loss, 'val_acc': acc}

        # log val loss
        self.logger.experiment.add_scalar("Val/Loss",
                                         loss,
                                         self.current_epoch)

        # log val acc
        self.logger.experiment.add_scalar("Val/Acc",
                                          acc,
                                          self.current_epoch)

        return {**out, 'log': out}

    def test_epoch_end(self, outputs):
        """
        Called at the end of a test epoch. Accumulates epoch performance.

        :param outputs: model predictions
        :return: A dictionary containing performance measures
        """

        # accumulate loss
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        
        # accumulate accuracy
        acc = th.cat([o['acc'] for o in outputs], 0).mean()

        # accumulate true positives, true negatives,...
        tp = th.cat([o['tp'] for o in outputs], 0).sum()
        tn = th.cat([o['tn'] for o in outputs], 0).sum()
        fp = th.cat([o['fp'] for o in outputs], 0).sum()
        fn = th.cat([o['fn'] for o in outputs], 0).sum()

        # calculate precision
        precision = tp / (tp + fp)

        # calculate recall
        recall = tp / (tp + fn)

        # calculate f1
        f1 = 2 * (precision * recall) / (precision + recall)

        # put them together
        out = {'test_loss': loss,
               'test_acc': acc,
               'precision': precision,
               'recall': recall,
               'f1': f1}

        # log test loss
        self.logger.experiment.add_scalar("Test/Loss",
                                          loss,
                                          self.current_epoch)

        # log test accuracy
        self.logger.experiment.add_scalar("Test/Acc",
                                          acc,
                                          self.current_epoch)

        # log test precision
        self.logger.experiment.add_scalar("Test/Precision",
                                            precision,
                                            self.current_epoch)

        # log test recall
        self.logger.experiment.add_scalar("Test/Recall",
                                            recall,
                                            self.current_epoch)

        # log test F1 score
        self.logger.experiment.add_scalar("Test/F1",
                                            f1,
                                            self.current_epoch)

        return {**out, 'log': out}


    def train_dataloader(self):
        """
        The training dataloader. The training set is prepared here.

        :return: The dataset used in training.
        """

        return th.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        """
        The validation dataloader. The validation set is prepared here.

        :return: The dataset used in validation.
        """

        return th.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )

    def test_dataloader(self):
        """
        The test dataloader. The test set is prepared here.

        :return: The training dataset used in test.
        """

        return th.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )

    def configure_optimizers(self):
        """
        Configures the optimizer used in training.

        :return: The optimizer
        """
        return th.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )


def objective(trial: optuna.Trial):
    """
    The objective function used for hyperparameter optimization.

    :param trial: an optuna trial  
    :return: The validation loss logs
    """

    # define the hyperparameter search spaces
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.01, 0.99)
    batch_size = trial.suggest_int("batch_size", 4, 64)
    seq_length = trial.suggest_categorical("seq_length", [90, 100, 128])

    # print the trial currently running
    print("This Trial: ", "\n",
          "lr: ", lr, '\n',
          "momentum: ", momentum, '\n',
          "batch_size: ", batch_size, '\n',
          "seq_length: ", seq_length, '\n',)

    # define the pruner
    prune = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val_loss"
    )

    # define the Pytorch Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=f'logs/{FLAGS.mode}/{FLAGS.train_ds}/epochs={FLAGS.epochs}_n_trials={FLAGS.n_trials}',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        callbacks=[prune],
        enable_checkpointing=False
    )

    # define the model with the sampled hyperparameters
    model = UserClassifier(
        lr=lr,
        momentum=momentum,
        batch_size=batch_size,
        seq_length=seq_length
    )

    # define the logger
    trainer.logger.log_hyperparams({
        "lr": lr,
        "momentum": momentum,
        "batch_size": batch_size
    })

    #train the model
    trainer.fit(model)

    return trainer.callback_metrics["val_loss"].item()


def main(_):
    
    # make sure the user defined all necessary flags
    if FLAGS.train_ds == 'default_train':
        raise Exception("Please define the train dataset.")
    elif FLAGS.val_ds == 'default_val':
        raise Exception('Please define the validation dataset.')
    elif FLAGS.test_ds == 'default_test':
        raise Exception('Please define the test dataset.')
    elif FLAGS.mode == 'default_mode':
        raise Exception('Please define the mode.')

    # HYPERPARAMETER OPTIMIZATION
    if FLAGS.mode =='train':

        # make sure the user defined all necessary flags
        if FLAGS.n_trials == 0:
            raise Exception("Please define the number of trials.")

        # training and optimization
        pruner = optuna.pruners.HyperbandPruner(3, 30, 2)

        # define the optuna study
        study = optuna.create_study(direction="minimize", pruner=pruner)

        # run the optuna study
        study.optimize(objective, n_trials=FLAGS.n_trials, show_progress_bar=True)

        # print the results of the hyperparameter optimization
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    # TRAINING AND TESTING
    elif FLAGS.mode == 'test':

        # NOTE: DEFINE THE HYPERPARAMETERS used in the training and testing run here!
        lr=0.042889952348328146
        momentum=0.7008668763590266
        batch_size=21
        seq_length=100

        # define the model to train
        model = UserClassifier(
            lr=lr,
            momentum=momentum,
            batch_size=batch_size,
            seq_length=seq_length
        )

        # define the logger
        tb_logger = TensorBoardLogger(f'logs/{FLAGS.mode}/{FLAGS.train_ds}/',
                                      name=f"test={FLAGS.test_ds}_epochs={FLAGS.epochs}_lr={lr}_m={momentum}_bs={batch_size}_seq_l={seq_length}",
                                      version=0
                                      )

        # define the Pytorch Lightning Trainer
        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if th.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=10)],
            checkpoint_callback=False
        )

        # fit the model
        # NOTE: can be commented out if your intention is to only test the model on some test
        # dataset and the classification head weights are already stored under /logs/models/classifier/ .
        # below, switch off storing the classifier head as well if you do this.
        trainer.fit(model)

        # path for saving the whole model (BERTweet + Classification Head)
        model_save_path = f'logs/models/full/{FLAGS.mode}/{FLAGS.train_ds}_epochs={FLAGS.epochs}_lr={lr}_m={momentum}_bs={batch_size}_seq_l={seq_length}.ckpt'
        
        # path for saving the classifier only
        classifier_save_path = f'logs/models/classifier/{FLAGS.mode}/{FLAGS.train_ds}_epochs={FLAGS.epochs}_lr={lr}_m={momentum}_bs={batch_size}_seq_l={seq_length}.ckpt'

        # save the model
        # NOTE: switched off to save storage, as whole omodel is around 500 MB in size
        #trainer.save_checkpoint(model_save_path)

        # save the classifier only
        classifier_only = th.nn.Sequential(model.classifier)
        th.save(classifier_only, classifier_save_path)

        # load the classifier
        my_classifier = th.load(classifier_save_path)

        # assign trained and stored weights to model classification head 
        model.classifier = my_classifier

        #test the model
        trainer.test(model)

if __name__ == '__main__':
    app.run(main)

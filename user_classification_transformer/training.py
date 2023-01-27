"""
Training a BERT model for user classification
"""

# !/usr/bin/env python3

from absl import app, flags, logging

import sh

import torch as th
import pytorch_lightning as pl

import nlp
import transformers
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from transformers import RobertaForSequenceClassification
from datasets import ClassLabel, Value

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

from datetime import datetime
import optuna
import warnings

warnings.filterwarnings("ignore", category=LightningDeprecationWarning)

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 3, '')
#flags.DEFINE_integer('batch_size', 10, '')
#flags.DEFINE_float('lr', '1e-3', '')
#flags.DEFINE_float('momentum', '.9', '')
# flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_string('model', 'vinai/bertweet-base', '')
#flags.DEFINE_integer('seq_length', 200, '')
flags.DEFINE_string('train_ds', 'df_all_train', '')
flags.DEFINE_string('val_ds', 'df_all_val', '')
flags.DEFINE_string('test_ds', 'df_all_test', '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

# import IPython ; IPython.embed() ; exit(1)


class UserClassifier(pl.LightningModule):
    def __init__(self,
                 lr=None,
                 momentum=None,
                 batch_size=None,
                 seq_length=None):

        super().__init__()


        #my_vocab_size = transformers.AutoTokenizer.from_pretrained(FLAGS.model).vocab_size + 1


        #print("############")
        #print(my_vocab_size)


        self.model = transformers.RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=FLAGS.model)
        #self.model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base')

        self.save_hyperparameters()

        # Freeze the RoBERTa model
        for param in self.model.roberta.parameters():
            param.requires_grad = False

        # Unfreeze the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.loss = th.nn.CrossEntropyLoss(reduction='none')

        # define hyperparams
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.seq_length = seq_length

        # define an empty array in which to store the frequency of token lengths
        #self.token_lengths = [0] * 500


    def prepare_data(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model)
        #tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                x['description'],
                max_length=self.seq_length,
                padding='max_length',
                truncation=True
            )

            #self.token_lengths[len(x['input_ids'])] += 1
            return x

        def _prepare_ds(dataset_name):
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
        token_type_ids = th.zeros_like(input_ids)  # create a dummy tensor filled with 0
        mask = (input_ids != 1).float()

        #print(input_ids.shape)
        #print(input_ids.min().min())
        #print(input_ids.max().max())
        #print(transformers.AutoTokenizer.from_pretrained(FLAGS.model).decode(input_ids.max().max().item()))


        output = self.model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        logits = output.logits
        #logits = self.model(input_ids=input_ids, attention_mask=mask).logits
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        labels = batch['is_gen_pub'].long()
        loss = self.loss(logits, labels).mean()

        #self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.logger.experiment.add_scalar("Train/Loss",
                                          loss,
                                          self.current_epoch)


        return {'loss': loss, 'log': {'train_loss': loss}}



    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        labels = batch['is_gen_pub'].long()
        loss = self.loss(logits, labels)
        acc = logits.argmax(-1) == labels
        acc = acc.float()

        #TODO
        self.log("val_loss", loss, on_step=False, on_epoch=True)


        #self.log("val/loss_epoch", loss.mean(), on_step=False, on_epoch=True)
        #self.log("val/acc_epoch", acc.mean(), on_step=False, on_epoch=True)

        return {'loss': loss, 'acc': acc}


    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        labels = batch['is_gen_pub'].long()
        loss = self.loss(logits, labels)
        acc = logits.argmax(-1) == labels
        acc = acc.float()
        #print('acc: ', acc)

        tp = (logits.argmax(-1) == labels) & (labels == 1)
        tp = tp.float()
        #print('true_positive: ', tp)

        tn = (logits.argmax(-1) == labels) & (labels == 0)
        tn = tn.float()
        #print('true_negative: ', tn)

        fp = (logits.argmax(-1) != labels) & (labels == 0)
        fp = fp.float()
        #print('false_positive: ', fp)

        fn = (logits.argmax(-1) != labels) & (labels == 1)
        fn = fn.float()
        #print('false_negative: ', fn)


        #self.log("test/loss_epoch", loss.mean())
        #self.log("test/acc_epoch", acc.mean())


        return {'loss': loss,
                'acc': acc,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn}


    def validation_epoch_end(self, outputs):

        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()



        out = {'val_loss': loss, 'val_acc': acc}




        self.logger.experiment.add_scalar("Val/Loss",
                                         loss,
                                         self.current_epoch)

        self.logger.experiment.add_scalar("Val/Acc",
                                          acc,
                                          self.current_epoch)

        return {**out, 'log': out}

    def test_epoch_end(self, outputs):
        print("Reached test_epoch_end)")
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()

        tp = th.cat([o['tp'] for o in outputs], 0).sum()
        tn = th.cat([o['tn'] for o in outputs], 0).sum()
        fp = th.cat([o['fp'] for o in outputs], 0).sum()
        fn = th.cat([o['fn'] for o in outputs], 0).sum()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)



        out = {'test_loss': loss,
               'test_acc': acc,
               'precision': precision,
               'recall': recall,
               'f1': f1}


        self.logger.experiment.add_scalar("Test/Loss",
                                          loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Test/Acc",
                                          acc,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Test/Precision",
                                            precision,
                                            self.current_epoch)

        self.logger.experiment.add_scalar("Test/Recall",
                                            recall,
                                            self.current_epoch)

        self.logger.experiment.add_scalar("Test/F1",
                                            f1,
                                            self.current_epoch)

        return {**out, 'log': out}


    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )

    def test_dataloader(self):
        return th.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )


def objective(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.01, 0.99)
    batch_size = trial.suggest_int("batch_size", 4, 64)
    #seq_length = trial.suggest_categorical("seq_length", [128, 256, 512])


    print("This Trial: ", "\n",
          "lr: ", lr, '\n',
          "momentum: ", momentum, '\n',
          "batch_size: ", batch_size, '\n')

    prune = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val_loss"
    )

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        callbacks=[prune]
    )

    model = UserClassifier(
        lr=lr,
        momentum=momentum,
        batch_size=batch_size,
        seq_length=100
    )

    trainer.logger.log_hyperparams({
        "lr": lr,
        "momentum": momentum,
        "batch_size": batch_size,
        "seq_length": 100
    })

    trainer.fit(model)



    return trainer.callback_metrics["val_loss"].item()


def main(_):
    if FLAGS.train_ds == 'default_train':
        raise Exception("Please define the train dataset.")
    elif FLAGS.val_ds == 'default_val':
        raise Exception('Please define the validation dataset.')
    elif FLAGS.test_ds == 'default_test':
        raise Exception('Please define the test dataset.')

    # training and optimization

    pruner = optuna.pruners.HyperbandPruner(3, 30, 2)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=6)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    '''
    model = UserClassifier()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string)
    #wandb_logger = WandbLogger(project="cac",
    #                           name=f"{FLAGS.train_ds}_{FLAGS.test_ds}_{dt_string}",
    #                           log_model=True)

    tb_logger = TensorBoardLogger('tb_logs/',
                                  name=f"{FLAGS.train_ds}_{FLAGS.test_ds}_{dt_string}",
                                  version=0
                                  )

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=tb_logger
    )

    trainer.fit(model)

    trainer.test(ckpt_path='best')

    #wandb.finish()
    
    '''


if __name__ == '__main__':
    app.run(main)



# TODO: Optional: setup this on a server (maybe not necessary due to short training times when only fine tuning classification head)
# How many epochs should I train for? --> Early stopping, then use best run for testing.
# TODO: Implement early stopping
# TODO: implement hyperparameter tuning while training(?)
# TODO: If training does not work, make hidden layer smaller? or remove it?
# TODO: implement train_opt and testing mode


'''
# Improvement Ideas:
# fine tune newly added special tokens.

In the Hugging Face transformers library, you can use the torch.nn.Embedding.from_pretrained() method to create an embedding layer with pre-trained weights, and then set the requires_grad attribute to False for the pre-trained weights while keeping it True for the newly added special token embeddings.

Here is an example on how to do this:

# Load pre-trained embeddings
embeddings = torch.nn.Embedding.from_pretrained(weights)

# Freeze pre-trained embeddings
embeddings.weight.requires_grad = False

# Add new special tokens to vocabulary
num_added_tokens = tokenizer.add_tokens(new_special_tokens)

# Fine-tune embeddings for new special tokens
embeddings.weight.data[-num_added_tokens:, :].requires_grad = True

This way, the pre-trained embeddings will not be fine-tuned during training, but the embeddings for the newly added special tokens will be fine-tuned.

'''
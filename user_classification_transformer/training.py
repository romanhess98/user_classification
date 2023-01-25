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
from datasets import ClassLabel, Value

import wandb
from pytorch_lightning.loggers import WandbLogger


flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', '1e-2', '')
flags.DEFINE_float('momentum', '.9', '')
# flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_string('model', 'vinai/bertweet-base', '')
flags.DEFINE_integer('seq_length', 20, '')
flags.DEFINE_string('train_ds', 'default_train', '')
flags.DEFINE_string('test_ds', 'default_test', '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

# import IPython ; IPython.embed() ; exit(1)

class UserClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.RobertaForSequenceClassification.from_pretrained(FLAGS.model)

        self.save_hyperparameters()


        # Freeze the RoBERTa model
        for param in self.model.roberta.parameters():
            param.requires_grad = False

        # Unfreeze the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)

        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                x['description'],
                max_length=FLAGS.seq_length,
                padding='max_length',
                truncation=True)
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

        self.train_ds, self.test_ds = map(_prepare_ds, (FLAGS.train_ds, FLAGS.test_ds))

    def forward(self, input_ids):
        mask = (input_ids != 1).float()
        logits = self.model(input_ids, mask).logits
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        labels = batch['is_gen_pub'].long()
        loss = self.loss(logits, labels).mean()

        self.log("train/loss", loss, on_step= False, on_epoch=True)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        labels = batch['is_gen_pub'].long()
        loss = self.loss(logits, labels)
        acc = logits.argmax(-1) == labels
        acc = acc.float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}

        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", acc, on_step=False, on_epoch=True)

        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.test_ds,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=4
        )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )


def main(_):
    if FLAGS.train_ds == 'default_train':
        raise Exception("Please define the train dataset.")
    elif FLAGS.test_ds == 'default_test':
        raise Exception('Please define the test dataset.')

    model = UserClassifier()

    #TO I need this? It said something about the dropout layers not being used otherwise
    #model.train()

    wandb_logger = WandbLogger(project="cac",
                               name=f"{FLAGS.train_ds}_{FLAGS.test_ds}",
                               log_model=True)

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=wandb_logger
    )
    trainer.fit(model)


if __name__ == '__main__':
    app.run(main)

# TODO: log accuracy, precision, recall, F1
# TODO: make sure everything is logged correctly
# TODO: find out maximum token number in all datasets, use this as maximum sequence length for all datasets
# TODO: setup this on a server (maybe not necessary due to short training times when only fine tuning classification head)
# TODO: How many epochs should I train for?

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
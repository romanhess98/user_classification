"""
Training a BERT model for user classification
"""

#!/usr/bin/env python3

from absl import app, flags, logging

import sh

import torch as th
import pytorch_lightning as pl

import nlp
import transformers
from datasets import ClassLabel, Value

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', '1e-2', '')
flags.DEFINE_float('momentum', '.9', '')
#flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_string('model', 'vinai/bertweet-base', '')
flags.DEFINE_integer('seq_length', 128, '')


FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


#import IPython ; IPython.embed() ; exit(1)


class UserClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')
    def prepare_data(self):
        #train_ds_old = nlp.load_dataset('imdb',
        #                                split='train[:5%]')


        #TODO: possible problem here, is_gen_pub is not type classLabel for some reason. Might cause problems later

        tokenizer = transformers.AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)
        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(
                x['description'],
                max_length=FLAGS.seq_length,
                padding='max_length',
                truncation=True)
            return x

        def _prepare_ds(split, k=0):
            if split == 'train':
                ds = nlp.load_dataset('csv',
                                      data_files='data/df_boston.csv',
                                      features=nlp.Features({'description': Value('string'),
                                                             'is_gen_pub': ClassLabel(num_classes=2),
                                                             'source': Value('string')}
                                                           ),
                                      split=f'train[:{k}%]+train[{k+10}%:]'
                                      )
            elif split == 'val':
                ds = nlp.load_dataset('csv',
                                      data_files='data/df_boston.csv',
                                      features=nlp.Features({'description': Value('string'),
                                                             'is_gen_pub': ClassLabel(num_classes=2),
                                                             'source': Value('string')}
                                                            ),
                                      split=f'train[{k}%:{k+10}%]')


            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'is_gen_pub'])
            return ds

        self.train_ds, self.val_ds = map(_prepare_ds, ('train', 'val'))

        #TODO: could later pass k as a parameter here and always return two datasets by default

    def forward(self, input_ids):
        #TODO: check how mask looks. Should be 0 for all padding tokens.
        mask = (input_ids != 1).float()
        logits, = self.model(input_ids, mask)

        print(mask)
        print(mask.shape)
        return logits

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        self.forward(batch['input_ids'])
    def velidation_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.val_ds,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False
        )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )


def main(_):
    model = UserClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug
        )
    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)


#TODO: in preparation, make labels to integers
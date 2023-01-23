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

flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_float('lr', '1e-2', '')
flags.DEFINE_float('momentum', '.9', '')
flags.DEFINE_string('model', 'bert-base-uncased', '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


#import IPython ; IPython.embed() ; exit(1)


class UserClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        #train_ds_old = nlp.load_dataset('imdb',
        #                                split='train[:5%]')
        train_ds = nlp.load_dataset('csv',
                                    data_files='data/df_boston.csv',
                                    features= nlp.Features({'description': Value('string'),
                                                            'is_gen_pub': ClassLabel(num_classes=2),
                                                            'source': Value('string')}
                                                           ),
                                    split='train'
                                    )

        #TODO: possible problem here, is_gen_pub is not type classLabel for some reason. Might cause problems later

        tokenizer = transformers.BertTokenizerFast.from_pretrained(FLAGS.model)

        #TODO: fix tokenizer: tokenizer.tokenize(train_ds[0]['description']) gives strange output
        # no cls and # repeatedly
        import IPython;
        IPython.embed();
        exit(1)

    def forward(selfself, batch):
        pass

    def training_step(selfself, batch, batch_idx):
        pass

    def train_dataloader(self):
        pass

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
        max_epochs=FLAGS.epochs
        )
    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)

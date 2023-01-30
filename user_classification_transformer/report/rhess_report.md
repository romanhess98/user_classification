---
title: "Improving the Classification of General Public and Institutional Twitter Users with Transformers."
bibliography: references.bib
format:
  html:
    code-fold: true
jupyter: python3
---


# Title:


# Introduction: brief explanation of the topic & its relevance plus the methodological approach

-	When using twitter data for research, important to differentiate between individual and organizational accounts
-	Present different definitions of individual and organizational
-	Present reasons why differentiating is important


Twitter is an interesting data source for computational communication scientists.
The Twitter API makes it comparably easy to obtain large amounts of text data.
This enables Big Data analyses for various purposes, but comes with its own challenges.
For example, Twitter users might not be representative of the general offline population.
Also, some twitter accounts might be fictious (cite Ahmed et al., 2017).
Especially the detection of bots has triggered a lot of research (e.g. cite Kudugunta, Feng et al., Shukla).
When researchers analyze Twitter data, they may want to exclude these cases from their datasets.
Or they might also want to isolate users from a certain country (cite Kwon) or gender (cite Vashtisth), depending on the research questions.

Another differentiation between Twitter users, which can potentially impact the findings of a given research question is the separation of general public and institutional users.
Institutional accounts differ from general public accounts in that they might represent a (governmental) institution, the media, or some other kind of organization.
Communication on Twitter can be social or non-social and spontaneous or strategic.
When researchers are mainly interested in natural communication between individuals, they might want to filter out the institutional users first (cite Kwon).
This filtering can also be helpful for recommendation engines, products opinion mining tools etc. (cite Kheir).
As institutional accounts make up 9% of Twitter accounts (cite McCorriston) they can skew the tweet sample significantly.

Defining who is a general public and who is an institutional user is not trivial and different approaches exist in the literature.
Li et al (cite) aimed at differentiating between male, female and *brand-related* Twitter users.
Yan et al. (cite) differentiate between *open* and *closed* accounts, where open accounts publish information to the public, with their goal being to promote products, services or themselves.
Closed accounts on the other hand publish information about their daily lives and use Twitter to communicate with their friends.
McCorriston et al. (cite)  created a tool to distinguish between organizational and individual accounts but did not define their definition of *organizational* in detail.
Lastly, De Choudhury et al. (cite) split Twitter users into ordinary individuals, organizations, and journalists/media bloggers.
The existing diversity and intransparency in defining user categories is not ideal.
There is potential for a unified system of deciding which users belong to which group to make it easier for researchers to filter out institutional users and make research more comparable.

Some differences between institutional and general public users have been highlighted by Kwon et al(cite).
In their study, general public users showed more retweeting than institutional users, less analytic language use and more affective language use.

Overall, for the above mentioned reasons, there is a need for reliable classification models, allowing researchers to filter out institutional accounts when their target sample should only contain general public users.
Several techniques for this purpose have already been introduced. They will be presented in the next section.


# Literature review: (short) review of existing literature incl. derivation and formulation of hypotheses/research questions

- Present different existing approaches for differentiating between individual and organizational accounts

- Present problem: even though many of them use text data, no transformer models used! No dense text embeddings.

- Present paper, stressing the advantages of transformer models in social sciences text analysis

- Present hypotheses:

    - Transformers achieve better performance within the datasets
    - Transformers achieve better performances across datasets (generalization)



- TWIROLE:
    - simple text feature
    - picture etc additionally

    - name, description, Twitter Follower-Friend Ratio, profile image, tweet
    - hybrid classifier
    - text features are rather simple (e.g. frequency based)
    - profile images: Convolutional Neural Network

Li et al. (cite) introduced TWIROLE, which uses Twitter users' name, description, the Follower-Friend Ratio, profile image and tweets as features for a hybrid classifier.
The text features are rather simple and frequency based in this case.



- Mc Corriston et al. (cite)
    - post content features, stylistic features, structural and behavioral features.
    - text features mostly frequency based, otherwise numerical features, fed into a support vector machine

McCorriston et al. (cite) used post content, stylistic as well as structural and behavioral features for their classifier.
The text features they implemented were mostly frequency-based, the other features were numerical.
All features were then fed into a support vector machine for classification

- Kheir et al (cite)
    - did not ue textual features at all, solely statistical-based on features such as number of followers, liked tweets, posts per day etc.

Kheir et al. (cite) did not use textual features at all.
Their features were solely statistical-based and included measures like the number of followers, liked tweets, posts per day etc.


- Kwon et al. (cite)
    - solely based on twitter profile descriptions
    - used the sci-kit learn package in python. Their representation of each profile text only considered the 500 most frequent terms
    - Thus the representation of each profile text was a 500 dimensional sparse vector containing the raw frequencies for the top 500 words.
    - Then a random forrest classifier was used to separate general public profiles from institutional accounts.

Kwon et al. (cite) trained a classifier solely based on twitter profile descriptions.
They used the sci-kit learn package in python and their representations of each profile text only considered the 500 most frequent terms in the data.
Thus the representation of each profile text was a 500 dimensional sparse vector containing the raw frequencies for the top 500 most frequent words.
A random forrest classifier was then used to separate general public profiles from institutional accounts.

None of the approaches we could find used dense (pretrained) text embeddings or transformers.
However, using such techniques yields high potential when working with natural language, as can be found in Twitter data.

Wankmüller (cite) illustrated this potential of Neural Transfer Learning using transformers for text analysis in the social sciences.
Transfer learning refers to a setting where something that has been learned in one situation is exploited in another situation. (cite Goodfellow p. 538)
For Transformer models such as BERT or RoBERTa, pretrained weights and token embeddings are freely available online.
These models have been trained with enormous amounts of data on well-designed pretraining tasks.
Researchers can use these pretrained encoders to obtain dense representations of their text data.
These features usually contain far more information than conventional, more simple text representations, as they use the knowledge inherent in the huge text corpora used for pretraining the transformer models.
They can then be fed into a smaller model which is then trained on the actual task the researcher is interested in.

This can improve the prediction performance on different NLP tasks (cite Wankmüller).

## Goal of the study:

We believe that this also holds for the task of Twitter user classification into general public and institutional users.
When text data is available, it is reasonable to expect that the usage of pretrained word embeddings and encoders can leverage performance.
Using a pretrained BERT model is especially promising in our case, because of the availability of BERTweet, which is based on the RoBERTa pre-training procedure.
It has been trained on 850 million english tweets and consists of 135 million parameters (cite Dat).

The goal of this work is to show the advantages of transformer models over conventional machine learning techniques in user classification.
In order to do that we tried to locate the datasets used by previous classification models and surpass the original authors' performance by using BERTweet for generating text features.
The only dataset we could find was that used by Kwon et al (cite), who used Twitter profile descriptions as the sole input for their classifier.
To be precise, the authors used five different datasets which were collected in the context of different events (*boston*, *brussels*, *mesa*, *quebec*, *random*).
They trained one model per dataset and then evaluated the performances within the same dataset as well as the classifiers' generalization across the other datasets.

## Hypotheses

As the profile descriptions are the only input used, we expect BERTweet to enable a better performance than in the original study.
Thus, we define our hypotheses as follows:

By designing our own classifier, using the BERTweet encoder to generate dense and meaningful representations of Twitter profile descriptions, we expect to achieve:

H1.1: a better performance on the *boston* test set, when trained on the *boston* training set than in the original study.

H1.2: a better performance on all other test sets, when trained on the *boston* training set than in the original study.

H2.1: a better performance on the *brussels* test set, when trained on the *brussels* training set than in the original study.

H2.2: a better performance on all other test sets, when trained on the *brussels* training set than in the original study.

H3.1: a better performance on the *quebec* test set, when trained on the *quebec* training set than in the original study.

H3.2: a better performance on all other test sets, when trained on the *quebec* training set than in the original study.

H4.1: a better performance on the *random* test set, when trained on the *random* training set than in the original study.

H4.2: a better performance on all other test sets, when trained on the *random* training set than in the original study.


# Method: description of the data set (origins & structure of the data, sampling approach, data preprocessing) and the analysis logic (incl. detailed explanation of used software packages & functions/models)

- describe datasets
- describe preprocessing
- describe tokenization
- describe RoBERTa model
- describe training setup

## Data
The dataset used by Kwon et al. (cite) consisted of three variables.
The first column contained the Twitter profile descriptions as strings.
The second column *is_gen_pub* listed the labels (institutional=0, general public=1).
The third column contained the *source* and thus allowed to split the dataset into the five smaller datasets *boston*, *brussels*, *mesa*, *quebec*, and *random*.

Overall the dataset consisted of 8945 cases.
Out of those, 2000 belonged to the *boston* dataset, 2008 to *brussels*, 918 to *mesa*, 1998 to *quebec* and 2021 to *random*.
Unfortunately, we spotted an error in the mesa data.
All annotations carried the label 1, and there were no negative cases.
Thus, we decided, to remove the mesa data from our analysis.
This sparked doubt about the reliability of the other datasets as well.
However, for the other sources, the distribution of positive and negative cases matched the description of the authors making them seem reliable enough for us to analyze (cite Kwon).



## Preprocessing
In a first step, we cleaned the dataset. The labels were created through manual annotation by different raters.
Because of that, the dataset contained some duplicate cases, where the same profile description had received different labels.
As inconsistent labels can hurt machine learning, we decided to handle these cases as follows.
First we split the dataset into the five smaller datasets based on the *source* column.
Then, under the assumption that the majority label is the correct one, we replaced multiple occurrences of the same profile description with a single row containing the description and the majority class as label.
An equal amount of positive and negative ratings resulted in the assignment of the positive class (1).
The dataset sizes before and after cleaning are shown in Table X.

| Dataset | Before | After |
|:-------:|:-------|:------|
| boston  | 2000   | 1675  |
| brussels| 1997   | 1997  |
| quebec  | 1998   | 1751  |
| random  | 2021   | 2003  |

: Dataset sizes before and after cleaning


In a last step each of the five datasets was split into a training, validation, and test set.
In the original paper, Kwon et al. used an 80/20 split to create a training and test set (cite).
As using BERTweet requires defining some hyperparameters, we decided to create a validation set as well.
Thus, we used a split of 70/10/20 for the training, validation, and test set.
Before splitting the data, we randomly shuffled it, to avoid different label distributions between the sets.
It would have been advantageous if the train-test split from the original paper had been known to outrule any other reasons for the expected difference in performance when using BERTweet.
However, as the original study's authors used the scikit learn library for training (cite), it is likely that they shuffled the data as well before training.
Overall, as the shuffling is random, it should not make a real difference.


##Tokenization
BERTweet (cite) offers its own tokenizer.
It is recommended to use it as this is the tokenizer that was used during pretraining.
For example, the tokenizer would transform the input:

'Feminist. Proud liberal. Pro-love. Colin Morgan fan. Book lover.
History geek. Cat person. Chocolate eater. Apparently a snowflake. #TheResistance #ImpeachTrump'

into the following tokens:


'Femin@@', 'ist@@', '.', 'Proud', 'liber@@', 'al.', 'Pro-@@', 'love@@', '.', 'Colin', 'Morgan', 'fan@@', '.', 'Book',
'lo@@', 'ver@@', '.', 'History', 'gee@@', 'k@@', '.', 'Cat', 'person@@', '.', 'Chocolate', 'ea@@', 'ter@@', '.', 'Apparently',
 'a', 'snow@@', 'fla@@', 'ke@@', '.', '#TheResistance', '#ImpeachTrump'.

Tokens in this case are not always whole words but smaller parts of words.
A syllable following another inside the same word is indicated by an *@*-symbol.
Inputs not understood by the tokenizer receive a special *unknown*-token.
When using BERTweet, one has to decide on a maximum input length to feed to the model.
The largest length (meaning the number of tokens) the model can take is 128.
The distribution of the number of tokens per profile description in all five datasets is shown in Figure X.

TODO: format (size etc.)

![X](figs/token_lengths.png)

: Figure X

Using hyperparameter optimization we tried three different configurations of 90, 100 and 128 as the maximum sequence length.
Any profile description longer than this would be cut at the maximum length and every description shorter than this would be filled up to this maximum length using a special padding token.

## Model
As described above, the BERTweet pretrained transformer model was used to obtain a meaningful representation of the profile descriptions.
This representation was dense (meaning it consisted of float numbers) and 768-dimensional.

These features were then fed into a simple linear classifier with two output neurons and a bias.
The classifier thus consisted of 1537 parameters.
The outputs were fed as logit values into Pytorch's cross entropy loss function.
From there the loss was backpropagated only through the classifier.
The parameters of BERTweet remained unchanged throughout the training process.

## Training
For training, the Pytorch Lightning library was used together with huggingface's nlp and transformers libraries.
The creation of our study's results consisted of two parts.
First the model was trained on the training set.
During this, a hyperparameter search was used to find the ideal hyperparameter configuration.
The hyperparameters used and the corresponding search spaces are shown in Table X (TODO).

|    Parameter     | Search Space     |
|:----------------:|:-----------------|
|  Learning Rate   | [1e-5 ; 1e-1]    |
|     Momentum     | [0.01; 0.99]     |
|    Batch Size    | [4 ; 64]         |
| Sequence Length  | [90 , 100 , 128] |

: The parameters and their corresponding search spaces used in the HPO



For hyperparameter optimization, the *HyperbandPruner* from the optuna library was used, taking the validation loss as the evaluation criterion.
Due to computational limitations, the search went on for 10 trials and 40 epochs per dataset (*boston*, *brussels*, *mesa*, *quebec*, and *random*).
A more thorough investigation over more epochs and more trials would likely result in a more optimal configuration than the one we found.

The results of the hyperparameter search are shown in Table X.

| Dataset | Learning Rate | Momentum | Batch Size | Sequence Length |
|:-------:|:--------------|:---------|:-----------|:----------------|
| boston  | 0.0088        | 0.4863   | 33         | 100             |
| brussels| 0.0837        | 0.5388   | 60         | 100             |
| quebec  | 0.0646        | 0.6405   | 15         | 128             |
| random  | 0.0429        | 0.7009   | 21         | 100             |

: The optimal hyperparameters based on the HPO



Once the best hyperparameter configuration was identified, within each dataset, a model was trained on the training data and tested on the test data.
The maximum number of epochs for training was 500.
Also, an early stopping mechanism had been implemented.
The model would train until the validation loss did not increase anymore for ten epochs.
Then the model parametrization from the epoch with the lowest validation loss would be used for testing on the test dataset.

Our evaluation criteria for the test set were the same as in the original study.
We measures accuracy, precision, recall, and F1 score.
The results of the study are presented in the next section.






# Results: testing of hypotheses/answering of research questions (incl. data visualizations where applicable)

- present results (compared to original authors' results)


| Train   | Test     | A    | P     |    R   |    F1  |
|:-------:|:--------:|:----:|:-----:|:------:|:------:|
| boston  | boston   |      |       |        |        |
|         | brussels |      |       |        |        |
|         | quebec   |      |       |        |        |
|         | random   |      |       |        |        |
| brussels| boston   |      |       |        |        |
|         | brussels |      |       |        |        |
|         | quebec   |      |       |        |        |
|         | random   |      |       |        |        |
| mesa    | boston   |      |       |        |        |
|         | brussels |      |       |        |        |
|         | quebec   |      |       |        |        |
|         | random   |      |       |        |        |
| quebec  | boston   |      |       |        |        |
|         | brussels |      |       |        |        |
|         | quebec   |      |       |        |        |
|         | random   |      |       |        |        |
| random  | boston   |      |       |        |        |
|         | brussels |      |       |        |        |
|         | quebec   |      |       |        |        |
|         | random   |      |       |        |        |

: Our study's results


| Train   |   Test    | A    | P    | R    | F1   |
|:-------:|:---------:|:-----|:-----|:-----|:-----|
| boston  |  boston   | .830 | .836 | .877 | .856 |
|         | brussels  | .736 | .876 | .736 | .800 |
|         |  quebec   | .806 | .923 | .851 | .886 |
|         |  random   | .685 | .852 | .760 | .803 |
| brussels| brussels  | .828 | .874 | .889 | .881 |
|         |  boston   | .794 | .762 | .935 | .840 |
|         |  quebec   | .869 | .919 | .934 | .926 |
|         |  random   | .704 | .859 | .778 | .817 |
| quebec  |  quebec   | .889 | .914 | .966 | .939 |
|         | brussels  | .811 | .827 | .931 | .876 |
|         |  boston   | .770 | .732 | .950 | .827 |
|         |  random   | .792 | .848 | .919 | .882 |
| random  |  random   | .743 | .867 | .827 | .847 |
|         | brussels  | .616 | .751 | .714 | .732 |
|         |  boston   | .586 | .636 | .764 | .694 |
|         |  quebec   | .677 | .897 | .725 | .802 |

: The original study's results using a random forrest classifier (cite Kwon et al) TODO: fat print results where I am better



# Discussion: summary and interpretation of results, limitations of the methodological approach, outlook

- why did this work? what does it mean
- why did it not work?
- what could be improved?
- takeaway also for other classifiers incorporating more information

    - more thorough hyperparameter search
    - retrain top layers of bert model
    - try different tokenizers
    - try different optimizers
    - try stopword removal
    -

# Sources

All the data and code used to create this report can be found at TODO


#TODO: make code beautiful for github
#TODO: include comments
#TODO: update yml
#TODO: rename yml to env.yml
#TODO: create citations
#TODO: format report to make it look nice
#TODO: replace Table numbers

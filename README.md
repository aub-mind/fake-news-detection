# Code and HyperParameters for the QICC Fake News Competition

This repository contains source code to all the code used in the Paper [LINK(coming soon)] and in the [QICC fake news competition](https://sites.google.com/view/fakenews-contest) by Team FAR-NLP and Team AI Musketeers

## Setup
For the Transformer-Based code, we used using Google Colab with a GPU accelerated instance. All transformers are based on HuggingFace's implementation.

The code is left as is from the competition with minor editing.

## Contributors

- [Wissam Antoun](https://github.com/WissamAntoun) wfa07@mail.aub.edu (Transformers-based models for fake news and News Domain identification, and News Domain comparison models)
- [Fady Baly](https://github.com/fadybaly) fgb06@mail.aub.edu (Feature Based Fake News and News Domain Detection Models)
- [Rim Achour](https://github.com/RimeA) rga34@mail.aub.edu (Feature Based Fake News and News Domain Detection Models)
- [Amir Hussein](https://github.com/AmirHussein96) anh21@mail.aub.edu (Feature Based Fake News and News Domain Detection Models and feature importance)

## Hyper-Parameters

Table 1: Fake News Detection Hyper-Parameters

| Models   |      Hyper-Parameters      |
|----------|:-------------:|
| NB |  smoothing parameter=10 |
| SVM |    penalty parameter=21, kernel= RBF   |
| RF | estimators=271 |
| XGBoost | estimators= 10, learning rate=1, gamma=0.5 |
| mBERT-base, <br>XLNET-base, <br>RoBERTa-base | MAXSEQLN:128, <br>LR:2e-5, <br>BATCHSIZE:32, <br>EPOCHS: up to 5 |

Table 2: News Domain Detection Hyper-Parameters

| Models   |      Hyper-Parameters      |
|----------|:-------------:|
| TF-IDF |  1 to 4-gram and 1 to 6-gram |
| CNN* |    EMBSIZE:300,<br>2-stacked CNNs<br>256 kernels of size 5<br>64 kernels of size 5<br>dropout: 0.1<br>Epochs: max of 40   |
| 3CCNN* | EMBSIZE:300<br>512 kernels of size 3,4 and 5<br>dropout: 0.3<br>Epochs: max of 40 |
| LSTM* | EMBSIZE:300<br>Hidden size: 300<br>Dropout: 0.05<br>Epochs: max of 40 |
| GRU* | Same as LSTM |
| Bi-LSTM* | Same as LSTM |
| Bi-LSTM with attention* | Same as LSTM |
| Bi-LSTM with attention** | Same as LSTM |
| mBERT-base, <br>XLNET-base, <br>RoBERTa-base | MAXSEQLN:128, <br>LR:2e-5, <br>BATCHSIZE:32, <br>EPOCHS: up to 5 |
| RMDL | EMBSIZE:50<br>MAXSEQUENCELENGTH : 500<br>MAXNBWORDS : 5000<br>Combination of (10 DNNs,10 RNNs,10 CNNs)<br>epochs : 100 each<br>dnn: default parameters except for<br>maxnodesdnn : 512<br>rnn & cnn : default parameters<br>Adam optimizer<br>dropout : 0.07 |

## Files Description
* Transformers-Team_AI_Musketeers/
    - Data_preperation.ipynb: Notebook used for dataset preperation for the transformer based models
    - Fake_News_BERT_RoBERTa.ipynb and Fake_News_Model_XLNet.ipynb: Notebooks used for transformer model training on the Fake News Dataset on google colab.
    - Fake_News_Models_with_BERTViz.ipynb: Testing the Bert Visualization tool for better interpretability
    - News_Domain_BERT_RoBERTa_XLNet.ipynb: Notebook used training the transfomer models on the domain Identidication task on google colab.
    - News_Domain_ML.ipynb: Notebook used for to evaluate other models on the news domain identification task
    - modeling.py: Extended the functionality of the BERT, XLNET and RoBERTa to support multilabel classification
    - multiutils.py: Data preperation and creating the fine-tunnning data for the multilabel classification
    - utils.py: Data preperation and creating the fine-tunnning data for binary classification
* Feature_based-Team_FAR_NLP/
    * fake_news/
        - google_search.py: Google search API access code for search result extraction
        - preprocessing.py: Preprocessing source code the fake news arcticles
    * topic/
        - get_entities.py: Google cloud Language API access code for extracting the entities from the text
        - model.py: Pytorch-based Bi-LSTM with Attention Model
        - train_topic.py: Training script
* Twitter_bot.ipynb: Notebook with the preproccessing, and model used for the twitter bot detection task

## Paper:
Cite our paper as:

```
@INPROCEEDINGS{Antoun2020:State,
AUTHOR="Wissam Antoun and Fady Baly and Rim Achour and Amir Hussein and Hazem Hajj",
TITLE="State of the Art Models for Fake News Detection Tasks",
BOOKTITLE="2020 IEEE International Conference on Informatics, IoT, and Enabling
Technologies (ICIoT) (ICIoT'2020)",
ADDRESS=", Qatar",
DAYS=14,
MONTH=mar,
YEAR=2020,
}
```

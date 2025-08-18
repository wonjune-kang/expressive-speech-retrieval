# Expressive Speech Retrieval using Natural Language Descriptions of Speaking Style

### Wonjune Kang, Deb Roy

This repository contains code for training and using the expressive speech retrieval system described in our ASRU 2025 paper, [Expressive Speech Retrieval using Natural Language Descriptions of Speaking Style](https://arxiv.org/abs/2508.11187), implemented in PyTorch.

The system consists of a text encoder and a speech encoder, and is based on using a CLIP/CLAP-style contrastive loss to learn cross-modal embeddings between expressive speech and text queries describing speaking style (e.g., "Spoken in a frustrated tone."). At inference time, the speech encoder should be used to compute (and cache) speech style embeddings for all utterances in a target speech corpus. Then, given a user’s natural language prompt describing a desired speaking style, the text encoder can be used to generate a text embedding which behaves as a query. Retrieval can be performed by computing the cosine similarity between the text embedding and all cached speech embeddings, and returning the most similar utterances.

If you find this work or our code useful, please consider citing our paper:

```
@article{kang2025expressive,
  title={Expressive Speech Retrieval using Natural Language Descriptions of Speaking Style},
  author={Kang, Wonjune and Roy, Deb},
  journal={arXiv preprint arXiv:2508.11187},
  year={2025}
}
```

## Prerequisites

You can install dependencies by running

```
pip install -r requirements.txt
```

## Pre-trained model weights and pre-processed data

The pre-trained speech and text encoder checkpoints for the [BERT, RoBERTa, T5, Flan-T5] + emotion2vec model configurations described in our paper can be found in the following Google Drive link. For convenience, we also provide the pre-processed data for IEMOCAP, ESD, and Expresso that we use in our experiments.

**[Google Drive Link](https://drive.google.com/drive/folders/1eJby_JvnC3-_SZxWrc4lrnpSsGsaytiv?usp=sharing)**

## Training a model

### Data preprocessing

You can prepare the data yourself by running the scripts provided in ```preprocess_data```. First, download the [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) and [Expresso](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset) datasets to a local directory. Then, run each of ```preprocess_iemocap.py```, ```preprocess_esd.py```, and ```preprocess_expresso.py```; make sure to set the paths in ```preprocess_esd.py``` and ```preprocess_expresso.py``` to where you saved your own version of each dataset (```preprocess_iemocap.py``` directly downloads the IEMOCAP dataset from Hugging Face Hub).

### Running training

You can train a model using ```train.py```, specifying a config file (```-c```), GPU index (```-g```), and run name (```-n```). Optionally, training can also be continued from a checkpoint using the ```-p``` flag.

```
python train.py \
  -c config/roberta_emotion2vec.yaml \
  -g 0 \
  -n roberta_emotion2vec \
  -p /path/to/checkpoint/checkpoint.pt
```

Make sure to change the dataset paths in the config file to match where you saved your own data locally, and set your own checkpoint logging directory. **Note that the code currently only supports training on a single GPU.** All experiments described in the paper were done using a single NVIDIA A6000 Ada GPU; to run experiments using the provided config files, **you will need at least 48GB of GPU memory**.

## Inference / Evaluation

You can run ```create_retrieval_test_splits.py``` to create retrieval trials for evaluation as described in the paper, or you can use the pre-computed retrieval trials pickle files provided in ```test_retrieval_splits```. ```run_retrieval_eval.py``` can be used to run the evaluation pipeline used in our paper. For example, to evaluate the RoBERTa + emotion2vec model, you can run:

```
python run_retrieval_eval.py \
  -g 0 \
  -c config/roberta_emotion2vec.yaml \
  -p /path/to/checkpoint/roberta_emotion2vec.pt
```

This will compute and print per-class retrieval performance for ESD and Expresso. If using the provided retrieval trial pickle files and checkpoint, you should get:

```
...

Retrieval scores for ESD:

...

Average
Recall@1: 0.6026
Recall@5: 0.9338
Recall@10: 0.9660
Recall@20: 0.9760

...

Retrieval scores for Expresso:

...

Average
Recall@1: 0.8159
Recall@5: 0.9826
Recall@10: 0.9857
Recall@20: 0.9894
```

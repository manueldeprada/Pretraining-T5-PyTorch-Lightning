# Pretraining T5 using PyTorch Lightning
## Introduction
This repo reimplements the original T5 denoising pretrain objective, which to the best of my knowledge, is not avaliable outside the TF-Mesh environment. 

Given a dataset with plain, unsupervised text, we will pass that text through T5 asking the model to complete corrupted spans. The task, as shown in HuggingFace's T5 page, consist of taking the sentence:

`The cute dog walks in the park`

mask the input:

`The <extra_id_0> walks in <extra_id_1> park`

and calculate the targets:

`<extra_id_0> cute dog <extra_id_1> the <extra_id_2>`

This allows T5 to learn from the given corpus. In this repo, the CORD-19 dataset is used to pretrain T5 so it performs better in downstream tasks. Although this script may have some training defects compared to Google's original code, it is small, flexible, portable and written with PyTorch. In our experiments with COVID-19 related downstream tasks, this pretraining greatly improved performance.

## Pretrained version
A pretrained version of T5 on the CORD-19 dataset is available in HuggingFace: [https://huggingface.co/manueldeprada/t5-cord19](https://huggingface.co/manueldeprada/t5-cord19) 

## Instructions
1. Prepare your dataset. In the case of CORD-19, download the latest version and run **extract_cord19.py**.
2. Take the json file generated, and run **standarize_cord19.json**
3. Place the standarized file in this repo's main folder and run:
    ```
    python3 prepare_dataset.py --tokenizer-name t5-base --valid-size 0.2 --dumps-size 100 --mask-probability 0.15
    ```
4. Previous step creates the *dataset_cache* folder with the dataset prepared for training. To start training, run
    ```
    python3 pretrain.py --epochs 5 --dataset cord19 --batch-size 8 --grad-accu 2 --max-len 64 --num-gpus=2
    ```

## Files in this repo
The files are intended to be used in the same order.
- The **extract_cord19.py** file takes the original CORD-19 dataset, as downloaded from the official site, and extracts a json file consisting of title, abstract and text from readable papers.
- **standarize_cord19** takes the previous json file and makes a big pretraining text file that has just a sentence per line.
- **prepare_dataset.py** is where all the magic happens. This file takes any dataset file(s) with a sentence per line and efficiently, taking chunks of the file(s):
    1. Tokenizes the sentence using T5's SentencePiece tokenizer. This is, splits words as necessary and assigns an integer to every word.
    2. Checks that sentences are not too long. For example, T5-base has a sequence length of 512 tokens. If a sentence is too long, the script will search for a dot and split it before discarding it, since it is very probable that such a long sentence was not well tokenized upstream. Also discard any sentence very short or with a lot of unknown tokens inside.
    4. Generate input and target ids as shown in the introduction, masking tokens with a given probability.
    5. Save the (input, target) pairs in 100mb joblib files that can be loaded efficiently for training, with a meta.json file containing lengths, train-valid partitioning and other meta info.
  
  Usage is as follows:
  ```
  Usage: prepare_dataset.py [OPTIONS] [TOKENIZER_NAME] [VALID_SIZE] 
                          [DUMPS_SIZE] [MASK_PROBABILITY]

  This script preprocesses and tokenizes a standardized pretraining text
  Dataset (a file with a sentence in each line) into a set of tokenized
  files for training and validating the text2text model.

  Options:
  --tokenizer-name TEXT           T5 tokenizer used for token ids.  [default: t5-base]
  --valid-size FLOAT              Validation set size.  [default: 0.2]
  --dumps-size INTEGER            Size in MB for the dataset raw files.  [default: 100]
  --mask-probability FLOAT        Probability of masking a token in a
                                  sentence.  [default: 0.15]
  ```
- **Cord19Dataset.py** implements a torch Dataset using the underlying structure from the previous phase, lazy loading the joblib files as needed.
- **pretrain.py** and **t2t/__init__.py**. Previous files were written from scratch, but this two files are modified from  from the excelent [ceshine/finetuning-t5/paraphrase](https://github.com/ceshine/finetuning-t5/tree/master/paraphrase) repo. A huge thanks to him for open-sourcing such well-written scripts!

## TO-DOs

- It may be worth studying if the masking phase would be better done before tokenization. In the current implementation, it can lead to strange input-targets like this one:
  ```
  >>> tokenizer.tokenize("are grouped")
  ['▁are', '▁', 'grouped']
  
  input_ids=['▁are', '<extra_id_0>', 'grouped']
  target_ids=['<extra_id_0>', '▁', '<extra_id_1>']
  ```
- When loading the dataset for training, we are loading the chunk files as needed. It may be worth exploring the sequentialness of the SortishSampler for more efficient loading and also freeing loaded chunks that are not longer needed.
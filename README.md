# BertPretrainFinetune

This repository is the official implementation of our pre-trained model BERT, which consists of two pre-training tasks and a fine-tuning task.


## Requirements
* **python**3
* **TensorFlow**=1.14.0
* **numpy**=1.20.1
* **six**=1.15.0


## Data and model Structure
We have released the benchmark dataset that concerns four types of vulnerabilities, namely reentrancy, timestamp dependence, integer overflow/underflow, and dangerous delegatecall.

```shell
${BertPretrainFinetune}
├── data
│   ├── finetune
│   └── pretrain
├── feature
│   ├── input
│   └── output
└── models
    ├── finetune
    │   └── integeroverflow
    │   └── reentrancy
    │   └── timestamp
    │   └── delegatecall
    ├── pretrain
    └── tfrecord
```

* `data/finetune`: the corresponding bytecode (instructions) for BERT fine-tuning.
* `data/pretrain`: the corresponding bytecode (instructions) for BERT pre-training.
* `models/finetune`: the corresponding fine-tuned BERT of different vulnerabilities.
* `models/pretrain`: the corresponding pre-trained BERT of different vulnerabilities.
* `models/tfrecord`: the corresponding tfrecord for BERT pre-training.
* `feature/output`: the output features of bytecode (instructions).


## Bert Pretrain

Before pret-raining the BERT, we first need to generate `tfrecord` file by the following scripts.
```shell
python create_pretrain_data.py	
   --input_file=./data/pretrain/$PATH \
   --output_file=./models/tfrecord/$PATH \
   --vocab_file=./vocab.txt \
   --do_lower_case=True \
   --max_seq_length=64 \
   --max_predictions_per_seq=20 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=20 \
   --use_gpu=True
```
Here, we can obtain the `tfrecord` file in `./models/tfrecord/$PATH`.


Now, you can pre-train BERT by the following scripts.
```shell
python run_pretraining.py	
    --input_file=./models/tfrecord/$PATH \
    --output_dir=./models/pretrain/$PATH \
    --do_train=True \
    --do_eval=True \
    --init_checkpoint=./models/pretrain/$PATH \
    --bert_config_file=./bert_config.json \
    --train_batch_size=32 \
    --max_seq_length=64 \
    --max_predictions_per_seq=20 \
    --num_train_steps=2000 \
    --num_warmup_steps=20 \
    --learning_rate=2e-5 \
    --use_gpu=True
```

Here, we can obtain the pre-trained BERT in `./models/pretrain/$PATH`.


## Bert Fine-tuning

Since different smart contract vulnerabilities have distinct features and patterns, we need further to fine-tune the pre-trained Bert on different smart contract vulnerabilities, respectively. For example:
```shell
python run_finetune.py
    --task_name=Bnge \
    --do_train=true \
    --do_eval=true \
    --data_dir=./data/finetune/$PATH \
    --vocab_file=./vocab.txt \
    --bert_config_file=./bert_config.json \
    --init_checkpoint=./models/pretrain/$PATH \
    --max_seq_length=64 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --output_dir=./models/finetune/$PATH \
    --use_gpu=True
```
Here, we can obtain the fine-tuned Bert in `./models/finetune/$PATH`.
Note that, the "Bnge" task is a fine-tuning task. 


## Feature Extraction 

After obtaining the fine-tuned Bert, we can exploit this model to extract the corresponding bytecode feature by the following scripts.
```shell
python extract_features.py
    --input_file=./feature/input/$PATH \
    --output_file=./feature/output/$PATH \
    --vocab_file=./vocab.txt \
    --bert_config_file=./bert_config.json \
    --init_checkpoint=./models/finetune/$PATH \
    --layers=-1 \
    --max_seq_length=64 \
    --batch_size=8 \
    --use_gpu=True
```
Here, we can obtain the corresponding bytecode feature in `./feature/output/$PATH`.


## Reference
1. Devlin, Jacob and Chang, Ming-Wei and Lee, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019. 
[BERT](https://arxiv.org/abs/1810.04805)

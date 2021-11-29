# NLP-Performer
This repository contains an implementation of the [Performer](https://arxiv.org/abs/2009.14794) model for Transformer networks. Please note that it is still work-in-progress.

To train the model on the reddit jokes dataset, run
```
python train_reddit_jokes_sw_tf_ver2_performer.py
```
and run
```
python infer_reddit_jokes_sw_tf_ver2_performer.py
```
to perform inference. It can be extended to any NLP related pre-training task for Language Modeling.

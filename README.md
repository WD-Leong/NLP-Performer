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

## Extending to Long Sequences
To allow the model to be trained on long sequences with a modest compute resources, `tf_ver2_performer_v1.py` is introduced. Following ideas from [Transformers are RNNs](https://arxiv.org/abs/2006.16236), the long sequence is divided into windows, each of `window_len`, and training proceeds by accumulating the gradients from the loss function across all windows. Please note that this modification has not been checked to ensure correctness.

To train the model on the Movie Dialogue dataset, run
```
python train_movie_dialogue_sw_tf_ver2_performer_v1.py
```
and run
```
python infer_movie_dialogue_sw_tf_ver2_performer_v1.py
```
to perform inference.

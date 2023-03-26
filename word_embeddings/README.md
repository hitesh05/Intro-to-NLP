# README

In the given assignment, I have implemented frequency-based modelling approaches, such as Singular Value Decomposition (SVD), and comparing it with the embeddings obtained using one of the variants of Word2vec, such as CBOW implementation with Negative Sampling.

It is highly recommended you use the colab notebook referenced here because it requires atleast 9 GB of RAM and 6 GB of GPU to train it. The link here is:

[colab file](https://colab.research.google.com/drive/1uZqGcH_Y-HiiQDtLiY6HBQKKDFJft_IF?usp=sharing)
I have also attached it with the submission for your reference. 

The python file is also present. To run the file:
```
python3 A3.py
```
Be careful about the path where you are saving the model while loading it.

Here is the link to the 3 models trained i.e. SVD, and the cbow with negative sampling models:
[models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/hitesh_goel_research_iiit_ac_in/Ei3J2GZhJ4FGvnjJTnm_hkcBWalbC6oQyDvSrrvSejEBWA?e=vSVvRr)

Model 1 is an implementation of the model specified by [this document](https://jalammar.github.io/illustrated-word2vec/) by taking the dot product in batchnorm after summing them up.

Model 2: The second is an interesting approach. I have reduced the problem to a binary classification problem. It
involves training the embedding layer by first calculating embeddings and then putting three basic linear layers to output a
probability which denotes whether the sample is negative or positive. We have the actual labels of them and then can train
the binary classification model simply using Binary Cross Entropy Loss function.
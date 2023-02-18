# Language Modelling

### Contents of the directory
- language_model.py (Kneser Ney and Witten Bell algorithms)
- neural_language_model.py (LSTM)
- datasets
    - Pride and Prejudice
    - a.txt (tokenised Pride and Prejudice)
    - Ulyssys
    - u.txt (tokenised Ulyssys)
- scores
    - 12 text files containing average perplexity scores and perplexity scores for each sentence for the 3 models.
models
    - LM5.h5
    - LM6.h5

### Running the language models

*Kneser Ney:*
```
python3 language_model.py k [path to corpus]
```

*Witten Bell:*
```
python3 language_model.py w [path to corpus]
```

### Running the neural model

It can only run on your local machine if you have PyTorch and Tensorflow installed. Else, run it on your Ada account.

```
python3 neural_language_model.py [path to model.h5]
```

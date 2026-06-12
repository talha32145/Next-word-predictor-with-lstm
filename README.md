News Headline Generator using LSTM

## Overview

This project is a Deep Learning NLP application that generates newsstyle headlines using an LSTM (Long Short-Term Memory) neural network. The model learns patterns from a large collection of news headlines and predicts the next words based on a given input sequence.

The goal of this project was to understand the fundamentals of sequence modeling, text preprocessing, word embeddings and language generation using recurrent neural networks.

---

## Features

* News headline generation using LSTM
* Text preprocessing and tokenization
* Sequence padding for variable-length inputs
* Next-word prediction
* Streamlit web application
* Hugging Face deployment
* Model and tokenizer serialization for inference

---

## Dataset

Dataset: Australian Broadcasting Corporation (ABC) News Headlines Dataset

The dataset contains thousands of real news headlines covering:

* Politics
* Sports
* Business
* Economy
* International Affairs
* Technology
* Environment

The dataset was used to train a language model capable of learning headline structure and vocabulary patterns.

---

## Project Workflow

### 1. Data Preprocessing

* Converted text to lowercase
* Tokenized headlines using Keras Tokenizer
* Generated numerical sequences
* Applied sequence padding
* Created input-output pairs for next-word prediction

### 2. Model Architecture

```text
Input Sequence
      │
      ▼
Embedding Layer (100 Dimensions)
      │
      ▼
LSTM Layer (128 Units)
      │
      ▼
Dropout Layer
      │
      ▼
Dense Layer (Softmax)
      │
      ▼
Predicted Next Word
```

### Architecture Summary

* Embedding Layer
* LSTM Layer
* Dropout Layer
* Dense Output Layer (Softmax)

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Streamlit
* Hugging Face Spaces

---

## Training Configuration

| Parameter           | Value                    |
| ------------------- | ------------------------ |
| Vocabulary Size     | 7000                     |
| Embedding Dimension | 100                      |
| LSTM Units          | 128                      |
| Optimizer           | Adam                     |
| Loss Function       | Categorical Crossentropy |
| Epochs              | 50                       |
| Framework           | TensorFlow / Keras       |

---

## Sample Predictions

### Input

```text
people
```

### Output

```text
people trapped after m5 pile up
```

---

### Input

```text
aust navy divers
```

### Output

```text
aust navy divers find sea mines
```

---

### Input

```text
businesses
```

### Output

```text
businesses not expecting quick recovery
```

---

### Input

```text
football academy
```

### Output

```text
football academy seeks expansion funds
```

---

## Results

The model successfully learned:

* News headline structure
* Common language patterns
* Domain-specific vocabulary
* Entity-action-object relationships

Example learned patterns:

* "Police say ..."
* "Court hears ..."
* "Prime minister ..."
* "Businesses not expecting ..."

---

## Limitations

Although the model generates realistic headlines, it has some limitations:

* Repetition may occur in longer generations
* Coherence decreases as generated text length increases
* LSTM struggles with long range dependencies
* Output quality depends heavily on training data

These limitations are expected in traditional recurrent neural networks and motivate the use of more advanced architectures such as Attention Mechanisms and Transformers.

---

## Future Improvements

* Bidirectional LSTM
* Attention Mechanism
* Encoder-Decoder Architecture
* Seq2Seq Models
* Transformer-Based Language Models

---

## Deployment

The model is deployed using Streamlit on Hugging Face Spaces.

You can interact with the application through the deployed web interface.

---

## Key Learnings

This project helped me understand:

* NLP preprocessing pipelines
* Tokenization and sequence generation
* Word embeddings
* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory (LSTM) networks
* Language modeling fundamentals
* Model deployment with Streamlit and Hugging Face

---

## Author

Muhammad Talha

Deep Learning and Machine Learning Enthusiast

Currently exploring:

* Sequence-to-Sequence Models
* Attention Mechanisms
* Transformer Architectures

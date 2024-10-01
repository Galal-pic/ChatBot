# Chatbot Using PyTorch and NLTK

This repository contains a simple chatbot built using a neural network implemented in PyTorch, with natural language processing (NLP) tasks powered by NLTK. The chatbot is trained to understand intents and respond accordingly based on predefined patterns.

## Features
- **Natural Language Processing (NLP)**: Utilizes tokenization, stemming, and a bag-of-words model to preprocess input data.
- **Neural Network**: A simple feedforward neural network with one hidden layer, implemented in PyTorch.
- **Custom Dataset**: The chatbot is trained on a dataset of intents and patterns, which can be customized to suit different purposes.
- **Training**: Includes training code that saves the trained model for future inference.

## Project Structure


### Files Overview
1. **`model.py`**: Defines the neural network class `NeuralNet` that is used to train the chatbot on intent data.
2. **`nltk_utils.py`**: Contains helper functions for text preprocessing:
   - `tokenize(sentence)`: Splits a sentence into words.
   - `stem(token)`: Reduces a word to its root form using PorterStemmer.
   - `bag_of_words(tokenized_sentence, words)`: Creates a bag-of-words vector from a tokenized sentence.
3. **`train.py`**: The main script that handles:
   - Loading and preprocessing the training data (`intents.json`).
   - Training the neural network using PyTorch.
   - Saving the trained model to a file (`data.pth`).
4. **`intents.json`**: A customizable JSON file that contains the intents (tags), patterns (sample phrases), and responses.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NLTK
- NumPy

### Installation

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    ```

### Training the Chatbot

1. Customize the **`intents.json`** file with your own patterns, intents, and responses.
2. Run the training script:
    ```bash
    python train.py
    ```
   The model will train on the provided data, and after training, the model will be saved as `data.pth`.

### Using the Trained Chatbot

Once the model is trained, you can integrate the model with a chat interface to create an interactive chatbot. You can load the saved model and use it to predict intents based on user input.

### Example of Intent Format

The **`intents.json`** file follows this format:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "How are you?", "Is anyone there?"],
      "responses": ["Hello!", "Hi there!", "Greetings!", "Hi! How can I assist?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["Goodbye!", "See you soon!", "Have a great day!"]
    }
  ]
}


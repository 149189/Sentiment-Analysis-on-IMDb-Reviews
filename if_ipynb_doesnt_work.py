import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Reduce the dataset size for faster experimentation
small_train_dataset = dataset['train'].select(range(1000))  # Use only 1000 examples for training
small_test_dataset = dataset['test'].select(range(1000))    # Use only 1000 examples for testing

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess the data with reduced sequence length
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
small_test_dataset = small_test_dataset.map(preprocess_function, batched=True)

# Convert datasets to TensorFlow format with a larger batch size
small_train_dataset = small_train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=32
)

small_test_dataset = small_test_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=32
)

# Custom training loop with tracking
optimizer = Adam(learning_rate=3e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
accuracy_metric = SparseCategoricalAccuracy()

# Lists to store accuracy and loss
train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Training step
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for batch in small_train_dataset:
        input_ids = batch[0]["input_ids"]
        attention_mask = batch[0]["attention_mask"]
        labels = batch[1]

        with tf.GradientTape() as tape:
            logits = model(input_ids, attention_mask=attention_mask, training=True).logits
            loss = loss_fn(labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss_avg.update_state(loss)
        epoch_accuracy.update_state(labels, logits)
    
    # Log training results
    train_losses.append(epoch_loss_avg.result().numpy())
    train_accuracies.append(epoch_accuracy.result().numpy())
    print(f"Training loss: {train_losses[-1]}, accuracy: {train_accuracies[-1]}")
    
    # Validation step
    val_loss_avg = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for batch in small_test_dataset:
        input_ids = batch[0]["input_ids"]
        attention_mask = batch[0]["attention_mask"]
        labels = batch[1]

        logits = model(input_ids, attention_mask=attention_mask, training=False).logits
        loss = loss_fn(labels, logits)

        val_loss_avg.update_state(loss)
        val_accuracy.update_state(labels, logits)
    
    # Log validation results
    val_losses.append(val_loss_avg.result().numpy())
    val_accuracies.append(val_accuracy.result().numpy())
    print(f"Validation loss: {val_losses[-1]}, accuracy: {val_accuracies[-1]}")

# Function to visualize predictions
def predict_sentiments_visual(text_data, model, tokenizer):
    # Initialize lists to store results
    sentiments = []
    probabilities_list = []
    
    # Iterate over each review in the text_data
    for review_text in text_data:
        # Preprocess the input review
        inputs = tokenizer(review_text, padding='max_length', truncation=True, max_length=64, return_tensors="tf")
        
        # Get model predictions
        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        predictions = tf.nn.softmax(logits, axis=-1)
        
        # Get the predicted class (0 or 1)
        predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
        
        # Map the predicted class to sentiment
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        
        # Append results to lists
        sentiments.append(sentiment)
        probabilities_list.append(predictions.numpy()[0])
    
    # Visualization
    num_reviews = len(text_data)
    x = np.arange(num_reviews)  # The label locations
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35  # Width of the bars
    
    # Create bars for each review's probabilities
    for i, (probabilities, sentiment) in enumerate(zip(probabilities_list, sentiments)):
        ax.bar(x[i] - bar_width / 2, probabilities[0], bar_width, label='Negative', color='red')
        ax.bar(x[i] + bar_width / 2, probabilities[1], bar_width, label='Positive', color='green')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Review Index')
    ax.set_ylabel('Probability')
    ax.set_title('Predicted Sentiments and Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Review {i+1}' for i in x])
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return sentiments, probabilities_list

# Example usage
text_data = [
    "This movie was absolutely amazing! The acting was superb and the plot was thrilling.",
    "The film was a total disappointment. The storyline was boring and predictable.",
    "An average movie with some entertaining moments, but it lacked depth."
]

sentiments, probabilities = predict_sentiments_visual(text_data, model, tokenizer)

for i, (review_text, sentiment) in enumerate(zip(text_data, sentiments)):
    print(f"Review {i+1}: {review_text}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"Probabilities: {probabilities[i]}")
    print()

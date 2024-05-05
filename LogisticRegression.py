import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_processing import train_images, train_labels, val_images, val_labels, test_images, test_labels
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images)
val_images_scaled = scaler.fit_transform(val_images)
test_images_scaled = scaler.fit_transform(test_images)
model = LogisticRegression(multi_class='ovr', max_iter=10000)



param_grid = {
    'C': [0.1, 1, 10],
}


best_val_accuracy = 0
best_params = {}
best_model = None

# Manually perform grid search to find the best parameters
for c in param_grid['C']:
        model = LogisticRegression(C=c, penalty='l2')
        print("going in")
        model.fit(train_images, train_labels)
        print("see you on the other side")
        train_predictions = model.predict(train_images_scaled)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_predictions = model.predict(val_images_scaled)
        val_accuracy = accuracy_score(val_labels, val_predictions)

        # Print the current configuration and accuracies
        print(f"Testing model with: C = {c}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Check if the current model is better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = {'C': c}
            best_model = model



# Get the best hyperparameters
print("Best Hyperparameters:", best_params)

# Training accuracy with the best model
train_predictions = best_model.predict(train_images_scaled)
train_accuracy = accuracy_score(train_labels, train_predictions)
print("Training Accuracy with Best Model:", train_accuracy)

# Validation accuracy with the best model
val_predictions = best_model.predict(val_images_scaled)
val_accuracy = accuracy_score(val_labels, val_predictions)
print("Validation Accuracy with Best Model:", val_accuracy)

# Test accuracy with the best model
test_predictions = best_model.predict(test_images_scaled)
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy with Best Model:", test_accuracy)
print("Test Classification Report:\n", classification_report(test_labels, test_predictions))
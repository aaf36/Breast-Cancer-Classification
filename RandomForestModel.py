from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data_processing import train_images, test_images, val_images, train_labels, test_labels, val_labels
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
model = RandomForestClassifier(random_state=42)
model.fit(train_images, train_labels)
y_pred = model.predict(val_images)
accuracy = accuracy_score(val_labels, y_pred)
print(f'Validation accuracy: {accuracy:.2f}')
# plot the learning curve of the trained model to examine bias and variance



train_sizes, train_scores, validation_scores = learning_curve(
    estimator=model,
    X=train_images,
    y=train_labels,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)


validation_mean = np.mean(validation_scores, axis=1)
validation_std = np.std(validation_scores, axis=1)


plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, validation_mean - validation_std,
                 validation_mean + validation_std, color='g', alpha=0.1)

plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, validation_mean, 'o-', color='g', label='Cross-validation score')

plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')

plt.show()

n_estimators_options = [50, 100, 200]
max_depth_options = [None, 10, 20, 30]
min_samples_split_options = [2, 5, 10]
min_samples_leaf_options = [1, 2, 4]


best_val_accuracy = 0
best_params = {}
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        for min_samples_split in min_samples_split_options:
            for min_samples_leaf in min_samples_leaf_options:
                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            random_state=42)
                rf.fit(train_images, train_labels)
                val_predictions = rf.predict(val_images)
                val_accuracy = accuracy_score(val_labels, val_predictions)
                print(f"Testing model with: n_estimators={n_estimators}, max_depth={max_depth}, "
                      f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }
                    best_model = rf


print(f"Best parameters found: {best_params}")
print(f"Best validation accuracy: {best_val_accuracy:.2f}")

test_predictions = best_model.predict(test_images)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

print('Classification Report:')
print(classification_report(test_labels, test_predictions))

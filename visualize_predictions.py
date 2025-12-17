import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------
# STEP 1: Load Test Data
# --------------------------
test_dir = r'C:\Users\cawle\Documents\brain tumor\Testing'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --------------------------
# STEP 2: Load Model and Predict
# --------------------------
model = load_model("best_model.h5")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# --------------------------
# STEP 3: Confusion Matrix
# --------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --------------------------
# STEP 4: Classification Report
# --------------------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# --------------------------
# STEP 5: Training History Plots
# --------------------------
try:
    with open('history.pkl', 'rb') as f:
        history = pickle.load(f)
    with open('fine_tune_history.pkl', 'rb') as f:
        fine_tune_history = pickle.load(f)

    acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    loss = history.history['loss'] + fine_tune_history.history['loss']
    val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.text(0.5, -0.15, 'a', ha='center', va='center', transform=plt.gca().transAxes)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.text(0.5, -0.15, 'b', ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig("accuracy_loss.png")
    plt.show()
except Exception as e:
    print("⚠️ Could not load history files. Error:", str(e))

# --------------------------
# STEP 6: Predicted Sample Images
# --------------------------
file_paths = test_generator.filepaths
inv_class_indices = {v: k for k, v in test_generator.class_indices.items()}
predicted_classes = y_pred

plt.figure(figsize=(15, 10))
for i in range(12):
    img = plt.imread(file_paths[i])
    true_label = inv_class_indices[y_true[i]]
    pred_label = inv_class_indices[predicted_classes[i]]

    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    plt.text(0.5, -0.15, f"{chr(97+i)}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

# --------------------------
# STEP 7: Augmented Sample Images
# --------------------------
train_dir = r'C:\Users\cawle\Documents\brain tumor\Training'
aug_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

aug_generator = aug_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=12,
    class_mode='categorical',
    subset='training'
)

aug_images, aug_labels = next(aug_generator)

plt.figure(figsize=(12, 8))
for i in range(12):
    label_index = np.argmax(aug_labels[i])
    label_name = inv_class_indices[label_index]

    plt.subplot(3, 4, i + 1)
    plt.imshow(aug_images[i])
    plt.title(f"Class: {label_name}")
    plt.text(0.5, -0.15, f"{chr(97+i)}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.axis('off')

plt.tight_layout()
plt.savefig("augmented_samples.png")
plt.show()

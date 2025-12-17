# ------------------------------------------
# Brain Tumor Classification - Full Pipeline (Improved Accuracy Version)
# ------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# PAGE 1: Data Preparation & Model Training
# ==========================================

# STEP 1: SET DATA PATHS
train_dir = r'C:\Users\cawle\Documents\brain tumor\Training'
test_dir = r'C:\Users\cawle\Documents\brain tumor\Testing'

# STEP 2: DATA AUGMENTATION & GENERATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# STEP 3: CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# STEP 4: BUILD THE MODEL (Xception)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# STEP 5: CALLBACKS
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)

# STEP 6: TRAINING PHASE 1
print("✅ Training starts now...")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint, lr_reduce],
    verbose=1
)

# Save history
with open('history.pkl', 'wb') as f:
    pickle.dump(history, f)

# STEP 7: FINE-TUNING PHASE 2
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
fine_tune_history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint, lr_reduce],
    verbose=1
)

# Save fine-tune history
with open('fine_tune_history.pkl', 'wb') as f:
    pickle.dump(fine_tune_history, f)

# ==========================================
# PAGE 2: Evaluation, Visualization & Plots
# ==========================================

# STEP 8: EVALUATE MODEL
model = load_model('best_model.h5')
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show(block=True)

# Final Accuracy
test_accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Final Test Accuracy: {test_accuracy * 100:.2f}%")

# STEP 9: PLOT ACCURACY & LOSS

def plot_history(history1, history2):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("accuracy_loss.png")
    plt.show(block=True)

plot_history(history, fine_tune_history)

# STEP 10: SAMPLE PREDICTION IMAGES
inv_class_indices = dict((v, k) for k, v in test_generator.class_indices.items())
file_paths = test_generator.filepaths

plt.figure(figsize=(15, 10))
for i in range(12):
    img = plt.imread(file_paths[i])
    true_label = inv_class_indices[y_true[i]]
    predicted_label = inv_class_indices[y_pred[i]]

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

# STEP 11: SHOW AUGMENTED TRAINING IMAGES
augmented_batch = next(train_generator)
images, labels = augmented_batch

plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i])
    plt.title(f"Class: {np.argmax(labels[i])}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("augmented_samples.png")
plt.show()
                    
# Model Inception Resnet v2

# Import libraries

import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                            classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(12049)

# --- Add Paths Here ---
OUTPUT_DIR = "/Users/leo/Desktop/BE Project/Model Training/Inception_Resnet_v2/Inception_Resnet_v2_Code_o:p/Attempt"  
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# --- End Add Paths Here ---

def get_plot_loss_acc(model, model_name):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model.history.history["loss"])
    plt.plot(model.history.history["val_loss"])
    plt.title(f"{model_name} \n\n model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper right")

    plt.subplot(2, 1, 2)
    plt.plot(model.history.history["accuracy"])
    plt.plot(model.history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="lower right")

    plt.tight_layout()

    # --- Save Plot ---
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_loss_acc_plot.png"))
    plt.close(fig) #Close the figure to prevent display
    # --- End Save Plot ---

def compute_confusion_matrix(
    ytrue, ypred, class_names, model_name
):
    cm = confusion_matrix(
        y_true=ytrue.labels,
        y_pred=np.argmax(ypred, axis=1),
    )

    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # --- Save Confusion Matrix ---
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close() #Close the figure to prevent display
    # --- End Save Confusion Matrix ---


def get_evaluate(data, name, model, class_names):
    score_model = model.evaluate(data, verbose=1)
    print(f"{name} loss: {score_model[0]:.2f}")
    print(f"{name} accuracy: {score_model[1]:.2f}")

    y_true = data.labels
    y_pred_raw = model.predict(data)
    y_pred = np.argmax(y_pred_raw, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClass-wise Accuracy (Precision):")
    class_precisions = {}
    for class_name in class_names:
        precision = report[class_name]['precision'] # Using precision as a proxy for class-wise accuracy here
        class_precisions[class_name] = precision
        print(f"{class_name}: {precision:.2f}")

    # --- Save Class-wise Accuracy Plot ---
    plt.figure(figsize=(8, 6))
    plt.bar(class_precisions.keys(), class_precisions.values(), color='skyblue')
    plt.ylabel("Precision")
    plt.xlabel("Class")
    plt.title(f"Class-wise Precision - {name}")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_{name}_classwise_accuracy.png"))
    plt.close()
    # --- End Save Class-wise Accuracy Plot ---

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def get_predict(data, model):  
    predict_model = model.predict(data)
    return predict_model

def get_metrics(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Accuracy Score - {model_name}: {acc:.2f}")
    print(f"Balanced Accuracy Score - {model_name}: {bal_acc:.2f}")
    print("\n")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Save Overall Metrics Plot ---
    metrics = {"Accuracy": acc, "Balanced Accuracy": bal_acc}
    plt.figure(figsize=(6, 5))
    plt.bar(metrics.keys(), metrics.values(), color='lightcoral')
    plt.ylabel("Score")
    plt.title(f"Overall Metrics - {model_name}")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_overall_metrics.png"))
    plt.close()
    # --- End Save Overall Metrics Plot ---

# Load data

base_dir = "/Users/leo/Desktop/BE Project/dataset/Git/dataset" # Path to your original dataset
#augmented_base_dir = "/Users/leo/Documents/BE_RIT/BE/Semester 8/Project - Phase 2/DataSet/Aug/augmented output"  # Path to your augmented dataset

train_path = os.path.join(base_dir, 'train')
valid_path = os.path.join(base_dir, 'val')
test_path = os.path.join(base_dir, 'test')

'''train_path = os.path.join(augmented_base_dir, 'train')
valid_path = os.path.join(augmented_base_dir, 'val')
test_path = os.path.join(base_dir, 'test')'''

# Definitions

model_name = "Inception ResNet V2"
class_names = ['Healthy', 'Doubtful', 'Minimal', 'Moderate', 'Severe']

target_size = (224, 224)
epochs = 50
batch_size = 16
img_shape = (224, 224, 3)

# Save model
save_model_ft = os.path.join(OUTPUT_DIR, f'model_{model_name}_ft.hdf5')

# Image data generator

aug_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
    horizontal_flip=True,
    brightness_range=[0.3, 0.8],
    width_shift_range=[-50, 0, 50, 30, -30],
    zoom_range=0.1,
    fill_mode="nearest",
)

noaug_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
)

train_generator = aug_datagen.flow_from_directory(
    train_path, class_mode="categorical", target_size=target_size, shuffle=True
)

valid_generator = noaug_datagen.flow_from_directory(
    valid_path,
    class_mode="categorical",
    target_size=target_size,
    shuffle=False,
)

y_train = train_generator.labels
y_val = valid_generator.labels

# Weight data

unique, counts = np.unique(y_train, return_counts=True)
print("Train: ", dict(zip(unique, counts)))

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
train_class_weights = dict(enumerate(class_weights))
print(train_class_weights)

# Train data

classes = np.unique(y_train)

# Callbacks
early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=8,
    restore_best_weights=True
)
plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, min_delta=0.01, 
    min_lr=1e-10, patience=4, mode='auto'
)

model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    input_shape=(img_shape),
    include_top=False,
    weights="imagenet",
)

# Fine-tuning
for layer in model.layers:
    layer.trainable = True

model_ft = tf.keras.models.Sequential(
    [
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation="softmax"),
    ]
)

model_ft.summary()

model_ft.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

start_ft = timeit.default_timer()

history = model_ft.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early, plateau],
    validation_data=valid_generator,
    class_weight=train_class_weights,
    verbose=1,
)

stop_ft = timeit.default_timer()

execution_time_ft = (stop_ft - start_ft) / 60
print(
    f"Model {model_name} fine tuning executed in {execution_time_ft:.2f} minutes"
)

model_ft.save(save_model_ft)

get_plot_loss_acc(model_ft, model_name)  # Save plot
get_evaluate(valid_generator, "Validation", model_ft, class_names)  # Save evaluation with class-wise accuracy

# Prediction 
y_pred_val = get_predict(valid_generator, model_ft)
get_metrics(y_val, np.argmax(y_pred_val, axis=1), model_name)  # Save metrics for validation set
compute_confusion_matrix(valid_generator, y_pred_val, class_names, model_name)  # Save confusion matrix for validation set
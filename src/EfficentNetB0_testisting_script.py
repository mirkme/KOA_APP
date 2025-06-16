
# Test Model EfficientNetB0

# Import libraries

import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, Input
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)

np.random.seed(12049)

# --- Add Paths Here ---
OUTPUT_DIR = "/Users/leo/Desktop/BE Project/Model Training/Test_Model/EfficentNetB0_Test/EfficentNetB0_Test_o:p/Attempt"  
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# --- End Add Paths Here ---

def get_classes(data_path, classes, data):
    print(f"---- {data} ----")
    knee_severity = {}
    count = 0

    for i in range(len(classes)):
        imgs = os.listdir(os.path.join(data_path, str(i)))
        knee_severity[i] = imgs
        count += len(imgs)

    for k, v in knee_severity.items():
        print(
            f"Grade {k} - {classes[k]}: {len(v)} images, {round((len(v) * 100) / count, 2)}%"
        )

    return knee_severity

def compute_confusion_matrix(
    predictions, valid_generator, class_names, model_name
):
    cm = confusion_matrix(
        y_true=valid_generator.labels,
        y_pred=np.argmax(predictions, axis=1),
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

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    #jet = cm.get_cmap("jet")
    #jet = cm.get_cmap("jet", 256)
    jet = matplotlib.colormaps.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img

# Load data

base_dir = "/Users/leo/Desktop/BE Project/dataset/Git/dataset"
test_path = os.path.join(base_dir, 'test')

class_names = ['Healthy', 'Doubtful', 'Minimal', 'Moderate', 'Severe']

tests_data = get_classes(test_path, class_names, 'test')

# Image data generator

model_name = "EfficientNetB0"
batch_size = 256
target_size = (224, 224)

aug_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    horizontal_flip=True,
    brightness_range=[0.3, 0.8],
    width_shift_range=[-50, 0, 50, 30, -30],
    zoom_range=0.1,
    fill_mode="nearest",
)

noaug_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_generator = noaug_datagen.flow_from_directory(
    test_path,
    batch_size=batch_size,
    class_mode="categorical",
    target_size=target_size,
    shuffle=False,
)

y_test = test_generator.labels

# Load Model

efficientnet = tf.keras.models.load_model('/Users/leo/Desktop/BE Project/Model Training/EfficentNetB0/EfficentNetB0_o:p/Finetune/Attempt/model_EfficientNetB0_ft.keras')

# Test data

predictions_efficientnet = efficientnet.predict(test_generator)
score_efficientnet = efficientnet.evaluate(test_generator, verbose=1)
print('Test loss:', score_efficientnet[0])
print('Test acc:', score_efficientnet[1])

# Metrics

get_metrics(
    test_generator.labels,
    y_pred=np.argmax(predictions_efficientnet, axis=1),
    model_name=model_name,
)

compute_confusion_matrix(
    predictions_efficientnet,
    test_generator,
    class_names,
    f"{model_name} fine tuning",
)

# Results

# Model definition
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
output_tensor = layers.Dense(5, activation='softmax')(x)

efficientnet = models.Model(inputs=input_tensor, outputs=output_tensor)

# Grad-Cam

efficientnet.summary()

# Get the EfficientNetB0 base model from your full model
last_conv_layer = base_model.get_layer("top_conv")
'''grad_model = tf.keras.models.Model(
    [base_model.input],
    [last_conv_layer.output, efficientnet.output]
)'''
grad_model = tf.keras.models.Model(
    inputs=efficientnet.input,
    outputs=[last_conv_layer.output, efficientnet.output]
)


# model
efficientnet.layers[-1].activation = None

for k, v in tests_data.items():
    print(f"Test data - {class_names[k]}")
    plt.figure(figsize=(10, 28))
    for i in range(min(6, len(v))):
        img_path = os.path.join(test_path, str(k), v[i])
        # prepare image
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=target_size
        )
        img = tf.keras.preprocessing.image.img_to_array(img)

        img_aux = img.copy()
        img_array = np.expand_dims(img_aux, axis=0)
        img_array = np.float32(img_array)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        y_pred = efficientnet.predict(img_array, verbose=0)[0]

        heatmap = make_gradcam_heatmap(grad_model, img_array)
        image = save_and_display_gradcam(img, heatmap)

        plt.subplot(1, 6, 1 + i, xticks=[], yticks=[])
        plt.imshow(image)
        plt.title(
            f"True {class_names[k]}\nPred: {class_names[np.argmax(y_pred)]}"
        )
        plt.xlabel(
            "\n".join([f"{c} = {p:.2f}" for c, p in zip(class_names, y_pred)])
        )

     # Save figure instead of showing
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_gradcam_{class_names[k]}.png"))
    plt.close()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_dir = r'C:\Users\Ajiita Wijaksara\Documents\Tsflow\cats_vs_dogs\train'
val_dir = r'C:\Users\Ajiita Wijaksara\Documents\Tsflow\cats_vs_dogs\validation'
test_dir = r'C:\Users\Ajiita Wijaksara\Documents\Tsflow\cats_vs_dogs\test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (100,100),
    batch_size = 16,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size = (100,100),
    batch_size = 16,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (100,100),
    batch_size = 16,
    class_mode = 'binary'
)

model = models.Sequential([
    layers.Conv2D(16,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same',input_shape=(100,100,3)),
    
    layers.Conv2D(16,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'),
    layers.Conv2D(32,kernel_size=(1,1),strides=(1,1),activation='relu'),
    layers.Conv2D(32,kernel_size=(1,1),strides=(1,1),activation='relu'),
    layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'),
    layers.Conv2D(64,kernel_size=(1,1),strides=(1,1),activation='relu'),
    
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1,activation='sigmoid')
    
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples//train_generator.batch_size,
    epochs=30,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples//validation_generator.batch_size,
    callbacks=[early_stop]
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'TEST ACCURACY = {test_accuracy*100:.2f}%')

img_path = r'C:\Users\Ajiita Wijaksara\Documents\Tsflow\cats_vs_dogs\test\Dog\48.jpg'
img = image.load_img(img_path, target_size=(100,100))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = img_batch/255

layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)

activations = activation_model.predict(img_preprocessed)

for layer_index, layer_activation in enumerate(activations):
    num_filters = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    print(f"\nVisualisasi Aktivasi Layer Conv2D ke-{layer_index+1}")
    
    num_cols = min(num_filters, 8)
    
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 5))
    
    if num_cols == 1:
        axes = [axes]
    
    for i in range(num_cols):
        ax = axes[i]
        ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Filter {i+1}')
    plt.tight_layout()
    plt.show()

predict = model.predict(img_preprocessed)[0][0]
print(predict)
label = 'Anjing' if predict > 0.5 else 'Kucing'
print('Gambar Itu adalah ',label)
model.save("cnn_cats_vs_dogs.h5")
print("Model berhasil disimpan!")

plt.imshow(img)
plt.axis('off')
plt.show()

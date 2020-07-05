import tensorflow as tf
from tensorflow import keras

print('TensorFlow version: {}'.format(tf.__version__))

# ######################################## Prepare Datasets ###########################################################

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


# Total number of classes and their names used for training the sequential model
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
number_of_classes = 10

print('Train_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('Test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

# ########################### Creating Sequential Layers of CNN #######################################################
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3, strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(number_of_classes, activation=tf.nn.softmax, name='Softmax')
])

print("Summary of the Model")
model.summary()

print("Compiling Model")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Compiling Model Done")

# ############################################Creating Checkpoint####################################################
checkpoint_path = "checkpoint/modelcheckpoint.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=2)
#####################################################################################################################

print("Starting Model Training")
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images,test_labels), callbacks=[cp_callback])
print("Model Training Done")

print("Evaluating Model Accuracy:")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: {}'.format(test_acc))
print('Test loss: {}'.format(test_loss))

# Saving Model to Directory
# Create .PB file which is proto buffer
print('\nSaving Model To Directory path')
export_path = "model/"

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nModel Saved To:', export_path)


# ##############################Load Existing Saved Model ##########################################################
new_model = tf.keras.models.load_model('model/')

# Check its architecture
new_model.summary()
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


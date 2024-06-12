import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Display first 5 images from the dataset
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis('off')
plt.show()

# Data Preprocessing and Augmentation
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
datagen = ImageDataGenerator(rotation_range=15,width_shift_range=0.1,
                              height_shift_range=0.1,
                              horizontal_flip=True,
                              rescale=1./255.
                              )

datagen.fit(x_train)
batch_size = 10
augmented_images = next(datagen.flow(x_train, batch_size=batch_size))
# Display augmented images
rows = (batch_size + 4) // 5
plt.figure(figsize=(15, 3 * rows))
for i in range(batch_size):
    plt.subplot(rows, 5, i + 1)
    plt.imshow(augmented_images[i][:, :, 0], cmap='gray')
    plt.title(class_names[int(y_train[i])])
    plt.axis('off')
plt.show()

# # Sequential Model with Adam Optimizer (CNN Model)
# seqModel = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Dropout(0.2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#
# seqModel.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
#                  metrics=['accuracy'])
# seqHistory = seqModel.fit(datagen.flow(x_train, y_train, batch_size=20),epochs=30,
#                             validation_data=(x_test, y_test))
# # Evaluate the model on test data
# loss, accuracy = seqModel.evaluate(x_test, y_test, verbose=0)
# print("Sequential Test Accuracy:", accuracy)
# print("Sequential Test Loss:", loss)
# #PLOT ACCURACY
# plt.plot(seqHistory.history['accuracy'], label='Sequential Model accuracy')
# plt.plot(seqHistory.history['val_accuracy'], label='Sequential Model val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.title("Sequential Model Accuracy")
# plt.show()
# #PLOT LOSS
# plt.plot(seqHistory.history['loss'], label='Sequential Model loss')
# plt.plot(seqHistory.history['val_loss'], label='Sequential Model val_loss')
# plt.title("Sequential Model Loss")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.show()

#CNN
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001),
           input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
opt = Adam(learning_rate=0.001)
lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * 0.95)
cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

cnn_history = cnn_model.fit(datagen.flow(x_train, y_train, batch_size=64),
                             epochs=50,
                             validation_data=(x_test, y_test),
                             callbacks=[lr_scheduler])

print("CNN Model Accuracy:", cnn_history.history['val_accuracy'][-1])
loss, accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)
print("CNN Test Accuracy:", accuracy)
print("CNN Test Loss:", loss)

#PLOT ACCURACY
plt.plot(cnn_history.history['accuracy'], label='CNN accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("CNN Model Accuracy")
plt.show()

#PLOT LOSS
plt.plot(cnn_history.history['loss'], label='CNN loss')
plt.plot(cnn_history.history['val_loss'], label='CNN val_loss')
plt.title("CNN Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# # #RNN MODEL
# x_train_flat = x_train.reshape(x_train.shape[0], -1, 3)
# x_test_flat = x_test.reshape(x_test.shape[0], -1, 3)
# rnnModel = Sequential([
#     SimpleRNN(128, input_shape=x_train_flat.shape[1:]),
#     Dense(10, activation='softmax')
# ])
# # Compile the model
# rnnModel.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# rnnHistory = rnnModel.fit(x_train_flat, y_train, epochs=30, validation_data=(x_test_flat, y_test))
# print("Model Accuracy:", rnnHistory.history['val_accuracy'][-1])
#
# # PLOT ACCURACY
# plt.plot(rnnHistory.history['accuracy'], label='Training accuracy')
# plt.plot(rnnHistory.history['val_accuracy'], label='Validation accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("RNN Model Accuracy")
# plt.legend()
# plt.show()
#
# #PLOT LOSS
# plt.plot(rnnHistory.history['loss'], label='Training loss')
# plt.plot(rnnHistory.history['val_loss'], label='Validation loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title("RNN Model Loss")
# plt.legend()
# plt.show()
# #
#
# # VGG Model
# vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# for layer in vgg_model.layers:
#     layer.trainable = False
#     vgg_model_flatten = Flatten()(vgg_model.output)
#     vgg_model_output = Dense(10, activation='softmax')(vgg_model_flatten)
#     vgg_model_final = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model_output)
#     vgg_model_final.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
#                          metrics=['accuracy'])
#
# vgg_history = vgg_model_final.fit(datagen.flow(x_train, y_train, batch_size=35),epochs=30,
#                                   validation_data=(x_test, y_test))
#
# vgg_model_final.save("vgg16_model.h5")
#  #PLOT ACCURACY
# plt.plot(vgg_history.history['accuracy'], label='VGG accuracy')
# plt.plot(vgg_history.history['val_accuracy'], label='VGG val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.title("VGG Model Accuracy")
# plt.show()
#
# #PLOT LOSS
# plt.plot(vgg_history.history['loss'], label='VGG loss')
# plt.plot(vgg_history.history['val_loss'], label='VGG val_loss')
# plt.title("VGG Model Loss")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.show()
#
#
# #LSTM
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)
# # Reshape data for LSTM
# x_train_lstm = x_train.reshape(x_train.shape[0], -1, 3)  # Flatten each image into a sequence
# x_test_lstm = x_test.reshape(x_test.shape[0], -1, 3)
#
# # Split training data into training and validation sets
# x_train_lstm, x_val_lstm, y_train_lstm, y_val_lstm = train_test_split(x_train_lstm, y_train, test_size=0.2, random_state=42)
# model_lstm = Sequential([
#      LSTM(128, input_shape=x_train_lstm.shape[1:]),
#      Dense(10, activation='softmax')
# ])
# model_lstm.compile(optimizer='adam',loss='categorical_crossentropy',
#                     metrics=['accuracy'])
# # Train the model
# LSTMhistory = model_lstm.fit(x_train_lstm, y_train_lstm, epochs=30, batch_size=64, validation_data=(x_val_lstm, y_val_lstm))
#
# # Evaluate the model on test data
# loss, accuracy = model_lstm.evaluate(x_test_lstm, y_test, verbose=0)
# print("Test Accuracy:", accuracy)
# print("Test Loss:", loss)
# # Plot Accuracy and Loss
# plt.plot(LSTMhistory.history['accuracy'], label='Training accuracy')
# plt.plot(LSTMhistory.history['val_accuracy'], label='Validation accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("LSTM Model Accuracy")
# plt.legend()
# plt.show()
#
# plt.plot(LSTMhistory.history['loss'], label='Training loss')
# plt.plot(LSTMhistory.history['val_loss'], label='Validation loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title("LSTM Model Loss")
# plt.legend()
# plt.show()
#
#
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
# # Preprocess the data
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)
#
# # Data Augmentation
# datagen = ImageDataGenerator(rotation_range=15,
#                              width_shift_range=0.1,
#                              height_shift_range=0.1,
#                              horizontal_flip=True,
#                              rescale=1./255.,
#                              )
#
# # Define ResNet50 model
# resnet_model = Sequential()
# resnet_model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3)))
# resnet_model.add(GlobalAveragePooling2D())
# resnet_model.add(Dense(10, activation='softmax'))
#
# # Compile the model
# resnet_model.compile(optimizer='adam',loss='categorical_crossentropy',
#                      metrics=['accuracy'])
#
# # Train the model
# resnet_history = resnet_model.fit(datagen.flow(x_train, y_train, batch_size=64),
#                                   steps_per_epoch=len(x_train) / 64,  epochs=10,validation_data=(x_test, y_test))
#
# loss, accuracy = resnet_model.evaluate(x_test, y_test)
# print("Test Accuracy:", accuracy)
# print("Test Loss:", loss)
#
# # Plot accuracy and loss
# plt.plot(resnet_history.history['accuracy'], label='Training Accuracy')
# plt.plot(resnet_history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('ResNet-50 Model Accuracy')
# plt.show()
#
# plt.plot(resnet_history.history['loss'], label='Training Loss')
# plt.plot(resnet_history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('ResNet-50 Model Loss')
# plt.show()

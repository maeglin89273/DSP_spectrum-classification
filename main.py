import numpy as np
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix

DATA_DIR = 'sound_data'

train_x = np.load(DATA_DIR + '/sound_X_train.npy').reshape((-1, 32, 32, 1))
train_y = to_categorical(np.load(DATA_DIR + '/sound_y_train.npy'), 20)
val_x = np.load(DATA_DIR + '/sound_X_val.npy').reshape((-1, 32, 32, 1))
labeled_val_y = np.load(DATA_DIR + '/sound_y_val.npy')
val_y = to_categorical(labeled_val_y, 20)
test_x = np.load(DATA_DIR + '/sound_X_test.npy').reshape((-1, 32, 32, 1))



def multiscale_layer(input_layer):
    conv_7x7 = Conv2D(64, (7, 7), activation='relu', padding='same')(input_layer)
    conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    return concatenate([conv_3x3, conv_7x7], axis=3)

def inception_module(input_layer, depth, c_depth=32):
    conv_1x1 = Conv2D(c_depth, (1, 1), activation='relu', padding='same')(input_layer)
    conv_5x5 = Conv2D(depth, (5, 5), activation='relu', padding='same')(conv_1x1)
    conv_1x1 = Conv2D(c_depth, (1, 1), activation='relu', padding='same')(input_layer)
    conv_3x3 = Conv2D(depth, (3, 3), activation='relu', padding='same')(conv_1x1)
    maxpool_3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv_1x1 = Conv2D(depth, (1, 1), activation='relu', padding='same')(maxpool_3x3)
    return concatenate([conv_3x3, conv_5x5, conv_1x1], axis=3)

def plot_accuracy(history):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    xc = np.arange(len(val_acc))

    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)


    plt.show()

def plot_confusion_matrix(model, x, y):
    pred_y = np.argmax(model.predict(x), axis=1)
    conf_mat = confusion_matrix(y, pred_y)



    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.colorbar()
    tick_marks = np.arange(conf_mat.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

def plot_spectrogram(x, y, labels, instance_num, start=0):
    rows = len(labels)
    cols = instance_num
    idx = 1
    for target_label in labels:
        c = 0

        for i, label in enumerate(y[start:]):
            if label == target_label:
                plt.subplot(rows, cols, idx)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.squeeze(x[start + i]), cmap='gray')
                idx += 1
                c += 1
                if c == 1:
                    plt.ylabel(str(target_label))
                elif c == instance_num:
                    break


    plt.show()

# plot_spectrogram(val_x, labeled_val_y, [16, 17, 18, 19], 7, 50)

input_spectrum = Input(shape=(32, 32, 1))
multi = multiscale_layer(input_spectrum)
max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(multi)
incept_1 = inception_module(max_pool, 72, 64)
max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(incept_1)
incept_2 = inception_module(max_pool, 128, 96)
max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(incept_2)
output = Flatten()(max_pool)
output = Dense(512, activation='relu')(output)
output = Dense(20, activation='softmax')(output)

model = Model(inputs=input_spectrum, outputs=output)

print(model.summary())
adam = Adam()

model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=1, batch_size=256, validation_data=(val_x, val_y))

plot_accuracy(history)
plot_confusion_matrix(model, val_x, labeled_val_y)


pred_test_y = np.argmax(model.predict(test_x), axis=1)
np.save('results.npy', pred_test_y)



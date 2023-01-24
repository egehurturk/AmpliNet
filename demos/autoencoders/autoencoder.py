import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


class Autoencoder:
    '''
    Deep convolutional autoencoder architecture with 
    mirrored encoder and decoder components
    '''

    def __init__(self, name, input_shape, conv_filters, conv_kernels, conv_strides, code_size):
        self.name = name
        self.input_shape  = input_shape # [width, height, channels]
        self.conv_filters = conv_filters # [2, 4, 8] first conv has 2 filters, second 4, third 8
        self.conv_kernels = conv_kernels # [3, 5, 3] first conv has 3x3, second 5x5, third 3x3
        self.conv_strides = conv_strides # [1, 2, 2] first conv has 1x1, second 2x2, third 2x2
        self.code_size = code_size

        self.encoder = None
        self.decoder = None
        self.model = None
        self._model_input = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, lr=0.0001):
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        mse_loss = keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)
    
    def train(self, x_train, epochs, batch_size):
        # since autoencoders try to replicate the input, the output should be the same
        self.model.fit(x_train,
                       x_train,
                       batch_size = batch_size, 
                       epochs = epochs,
                       shuffle = True) 

    def reconstruct(self, image):
        latent_representations = self.encoder.predict(image)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def save(self, folder="."):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        parameters = [
            self.name,
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.code_size
        ]
        save_path = os.path.join(folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

        save_path = os.path.join(folder, "weights.h5")
        self.model.save_weights(save_path)
       
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)
        return autoencoder

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_in = self._model_input
        model_out = self.decoder(self.encoder(model_in))
        self.model = keras.Model(model_in, model_out, name = self.name)

    def _build_encoder(self):
        encoder_input_layer = self._add_encoder_input()
        encoder_conv_layers = self._add_conv_layers(encoder_input_layer)
        bottleneck = self._add_bottleneck(encoder_conv_layers)
        self._model_input = encoder_input_layer
        self.encoder = keras.Model(encoder_input_layer, bottleneck, name = "encoder")

    def _build_decoder(self):
        decoder_in = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_in)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_t_layers = self._add_conv_t_layers(reshape_layer)
        decoder_out = self._add_decoder_out(conv_t_layers)
        self.decoder = keras.Model(decoder_in, decoder_out, name = "decoder")

    def _add_decoder_input(self):
        return keras.layers.Input(shape = self.code_size, name = "decoder_input")

    def _add_dense_layer(self, decoder_in):
        num_neurons = np.prod(self._shape_before_bottleneck)
        return keras.layers.Dense(num_neurons, name = "decoder_dense")(decoder_in)

    def _add_reshape_layer(self, dense_layer):
        return keras.layers.Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_t_layers(self, layers):
        """
        Add conv transpose blocks
        """
        for layer_i in reversed(range(1, self._num_conv_layers)):
            layers = self._add_conv_t_layer(layer_i, layers)
        return layers

    def _add_conv_t_layer(self, i, layer):
        layer_num = self._num_conv_layers - i
        conv_t_layer = keras.layers.Conv2DTranspose(
            filters = self.conv_filters[i],
            kernel_size = self.conv_kernels[i],
            strides = self.conv_strides[i],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{layer_num}"
        )
        layer = conv_t_layer(layer)
        layer = keras.layers.ReLU(name=f"decoder_relu_{layer_num}")(layer)
        layer = keras.layers.BatchNormalization(name=f"decoder_bn_{layer_num}")(layer)
        return layer

    def _add_decoder_out(self, layer):
        conv_t_layer = keras.layers.Conv2DTranspose(
            filters = 1, # since spectograms can be interpreted as grayscale images
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        layer = conv_t_layer(layer)
        output_layer = keras.layers.Activation("sigmoid", name="sigmoid")(layer)
        return output_layer

    def _add_encoder_input(self):
        return keras.layers.Input(shape = self.input_shape, name = "encoder_input")

    def _add_conv_layers(self, encoder_input):
        """
        Create all convolutional blocks in encoder
        """
        x = encoder_input
        for layer_i in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_i, x)
        return x

    def _add_conv_layer(self, i, layer):
        """
        Adds a convolutional block to a of layers, consisting of conv2d + ReLU + batch norm
        """
        layer_no = i + 1
        conv = keras.layers.Conv2D(
            filters = self.conv_filters[i],
            kernel_size = self.conv_kernels[i],
            strides = self.conv_strides[i],
            padding = "same",
            name = f"encoder_conv_layer_{layer_no}"
        )
        layer = conv(layer)
        layer = keras.layers.ReLU(name=f"encoder_relu_{layer_no}")(layer)
        layer = keras.layers.BatchNormalization(name=f"encoder_bn_{layer_no}")(layer)
        return layer

    def _add_bottleneck(self, layer):
        """
        Flatten data and add bottleneck
        """
        self._shape_before_bottleneck = keras.backend.int_shape(layer)[1:] # [2, 7, 7, 32]
        layer = keras.layers.Flatten()(layer) 
        layer = keras.layers.Dense(self.code_size, name = "encoder_output")(layer)
        return layer
        
def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    X_train, y_train, x_test, y_test = load_mnist()


    LR = 0.0005
    BATCH_SIZE = 32
    EPOCHS = 20

    # ae = Autoencoder(
    #     "Autoencoder_EGE",
    #     input_shape=(28, 28, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernels=(3, 3, 3, 3), 
    #     conv_strides=(1, 2, 2, 1),
    #     code_size=2
    # )
    # ae.summary()
    # ae.compile(lr=LR)
    # ae.train(X_train[:10000], batch_size=BATCH_SIZE, epochs=EPOCHS)
    # ae.save("ae_model")

    ae = Autoencoder.load("ae_model")
    
    sample, labels = select_images(x_test, y_test, 12)
    print(sample.shape)

    re_im, lat_rep = ae.reconstruct(sample)
    plot_reconstructed_images(sample, re_im)
    plot_images_encoded_in_latent_space(lat_rep, labels)

    

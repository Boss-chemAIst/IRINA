import tensorflow
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np
import pickle

from Processing.mesh_preprocessing import x_test, x_train, y_test, y_train

latent_dim = 8


def sampling(args):
    z_mean_value, z_log_v = args
    batch = K.shape(z_mean_value)[0]
    dim = K.int_shape(z_mean_value)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean_value + K.exp(0.5 * z_log_v) * epsilon


image_size = x_train.shape[1]


input_img = Input(shape=(image_size, image_size, 1), )

h = Conv2D(16, kernel_size=3, activation='relu', padding='same', strides=2)(input_img)
enc_output = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=2)(h)

shape = K.int_shape(enc_output)
x = Flatten()(enc_output)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
dec_output = Conv2DTranspose(1, kernel_size=3, activation='relu', padding='same')(x)

decoder = Model(latent_inputs, dec_output, name='decoder')

outputs = decoder(encoder(input_img)[2])
vae = Model(input_img, outputs, name='vae')

reconstruction_loss = binary_crossentropy(K.flatten(input_img), K.flatten(outputs))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

vae.fit(x_train, epochs=20, batch_size=128, shuffle=True, validation_data=(x_test, None))


filename = 'autoencoder.sav'
enc = 'encoder.sav'
dec = 'decoder.sav'

pickle.dump(vae, open(filename, 'wb'))
pickle.dump(encoder, open(enc, 'wb'))
pickle.dump(decoder, open(dec, 'wb'))

'''
vae = pickle.load(open(filename, 'rb'))
encoder = pickle.load(open(enc, 'rb'))
decoder = pickle.load(open(dec, 'rb'))
'''

'''
z_mean, _, _ = encoder.predict(x_test)
decoded_imgs = decoder.predict(z_mean)

n = 10
plt.figure(figsize=(20, 4))
for i in range(10):
    plt.gray()
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i +1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

x = tensorflow.concat([x_train, x_test], 0)
y = tensorflow.concat([y_train, y_test], 0)

feat_extractor = Model(inputs=vae.input, outputs=vae.get_layer("dense").output)


zeros = [i for i in range(len(y.numpy())) if y.numpy()[i] == 0]
ones = [i for i in range(len(y.numpy())) if y.numpy()[i] == 1]

for idx in zeros:
  img = np.reshape(x[idx,:,:,:], (-1,28,28,1))
  pred = np.reshape(feat_extractor.predict(img)[0], (16,1)).T
  if idx == 1:
    zeros_list = pred
    continue
  zeros_list = np.concatenate((zeros_list, pred), axis=0)

for idx in ones:
  img = np.reshape(x[idx,:,:,:], (-1,28,28,1))
  pred = np.reshape(feat_extractor.predict(img)[0], (16,1)).T
  if idx == 3:
    ones_list = pred
    continue
  ones_list = np.concatenate((ones_list, pred), axis=0)

zeros_file = open("zeros.txt", "w")
for row in zeros_list:
    np.savetxt(zeros_file, row)

zeros_file.close()

ones_file = open("ones.txt", "w")
for row in ones_list:
    np.savetxt(ones_file, row)

ones_file.close()

'''
zeros_list = np.loadtxt("zeros.txt").reshape(6903, 16)
ones_list = np.loadtxt("ones.txt").reshape(7877, 16)

zeros_list = pd.DataFrame(zeros_list)
ones_list = pd.DataFrame(ones_list)
'''

'''
Outputs:
1) NxM dataframe (N genes of latent vector; M points or number of target meshes): to generate each gene value.
2) N vector (variances of N genes): to set the importance of each gene.
'''

vae_latent_vector_df = None
vae_latent_vector_length = 0

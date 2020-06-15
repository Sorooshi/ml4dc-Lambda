"""

This is an implementation of "Clustering the Latent/Reconstructed Data point By Variational Auto-Encoder" algorithm for
ML4DC  studies on CERN data set. These studies are devoted to the task of abnormality detection on
CERN's experimental results.
In these studies, Fedor Ratnikov is the Research Supervisor and Soroosh Shalileh is the Research Assistant.

In order to evaluate the performance of an algorithm, it is a common practice to run that algorithm
several times and compute the average and the standard deviation of the metric under consideration.
 Moreover, the supervised and semi-supervised algorithms need a training set, a validation set and a test set.

This program is written so that it satisfies all the aforementioned requirements.
Therefore, the program expects to receive the entire of a data set -- including the training sets,
validation sets and, test sets, with all the necessary repeats -- a .pickle format.

Concretely, Suppose we have a prepared data set in Numpy format X, and the corresponding labels Y.
At each repeat, we split X into x_train, y_train, x_validation and y_validation and, finally, x_test and y_test.
 and we store them in a dictionary such that keys represent the mentioned names and the values (of the dict)
 are holding the corresponding data splits.

## Input:
With such a setting, this program accepts a NAME.pickle data set. And one, simply, needs to enter the path of the stored data set via terminal.

## Run the program:
To run this program one needs to enter and modify the following command in the terminal:

python name_of_program.py  --Name='Name_of_dataset' --PreProcessing='z' --Run=1 --Path='Path_to_stored_dataset.pickle.(--N_epochs=100 if that algorithm has any epochs)

Where:

'--Path', type=str, Path to load the data sets;

'--Name', type=str, Name of the Experiment/Dataset;

'--Run', type=int, default=0, Whether to run the program or to evaluate the stored results'.
 Passing one to this argument will run the program and passing zero will read the stored data and
  print out the evaluation results.

'--PreProcessing', type=str, default='z', A string determining which preprocessing method should be applied.
 "z" represents Z-scoring, "rng" represents MinMax and 'NP" represents No Preprocessing.

'--N_epochs', type=int, default=500, An int. denoting the number of epochs.

NB: This program may accept more or fewer arguments.
For further information see the argument parsing section and the help of the arguments.

## Output:
NB: After running this program, the corresponding results of each data split repeat
will be automatically saved in the path which data set is stored.

NB: Several metrics are considered for evaluating the performance of an algorithm.
Once the running phase is done the test results will be print out on the screen.

NB: in some algorithms, the corresponding section for printing the validation results are commented,
in the case you need them please uncomment them.


- Prepared by: Soroosh Shalileh.

- Contact Info:
    In the case of having comments please feel free to send an email to sr.shalileh@gmail.com

"""

import os
import time
import pickle
import warnings
import argparse
import numpy as np
import tensorflow as tf
import ADmetrics as adm
from sklearn import metrics
from sklearn import mixture
import data_standardization as ds
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

BATCH_SIZE = 100
warnings.filterwarnings('ignore')
tf.keras.backend.set_floatx('float32')
CASES = ['original', 'reconstructed', 'latent']

MVB = True  # if True decoder is Multi-Variate Bernoulli else Multi-Variate Gaussian (MVG)


def args_parser(args):
    path = args.Path
    name = args.Name
    run = args.Run
    with_noise = args.With_noise
    pp = args.PreProcessing
    setting = args.Setting
    latent_dim_ratio = args.Latent_dim_ratio
    n_epochs = args.N_epochs

    return path, name, run, with_noise, pp, setting, latent_dim_ratio, n_epochs


class Sampling(tf.keras.layers.Layer):

    """Uses (z_mean, z_log_var) to sample z, the vector encoding a datapoint"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var) * epsilon  # tf.exp(0.5 * z_log_var)


class Encoder(tf.keras.layers.Layer):

    """Maps a datapoint vector to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim, intermediate_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj_1 = tf.keras.layers.Dense(int(intermediate_dim/2), activation='relu')
        self.dense_proj_2 = tf.keras.layers.Dense(int(intermediate_dim/2), activation='relu')
        self.dense_proj_3 = tf.keras.layers.Dense(int(intermediate_dim/4), activation='relu')
        self.dense_proj_4 = tf.keras.layers.Dense(int(intermediate_dim/8), activation='relu')
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj_1(inputs)
        x = self.dense_proj_2(x)
        x = self.dense_proj_3(x)
        x = self.dense_proj_4(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):

    """Converts z, the encoded datapoint vector, back into a original datapoint."""

    def __init__(self, original_dim, intermediate_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        # self.dense_proj_1 = tf.keras.layers.Dense(int(intermediate_dim/8), activation='relu')
        # self.dense_proj_2 = tf.keras.layers.Dense(int(intermediate_dim/4), activation='relu')
        # self.dense_proj_3 = tf.keras.layers.Dense(int(intermediate_dim/2), activation='relu')
        self.dense_proj_4 = tf.keras.layers.Dense(int(intermediate_dim/2), activation='relu')

        if MVB is True:
            self.dense_output = tf.keras.layers.Dense(original_dim, activation='sigmoid')
        else:
            self.dense_mean = tf.keras.layers.Dense(original_dim)
            self.dense_log_var = tf.keras.layers.Dense(original_dim)
            self.sampling = Sampling()

    def call(self, inputs):
        z = self.dense_proj_4(inputs)
        # z = self.dense_proj_2(z)
        # z = self.dense_proj_3(z)
        # z = self.dense_proj_4(z)
        if MVB is True:
            return self.dense_output(z)
        else:
            mu = self.dense_mean(z)
            log_var = self.dense_log_var(z)
            x_hat = self.sampling((mu, log_var))
            return x_hat


class VariationalAutoEncoder(tf.keras.Model):

    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim, latent_dim, name='v-auto-encoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim, )
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss
        # kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) (wrong!)
        kl_loss = - 0.5 * tf.reduce_mean(tf.square(z_log_var) - tf.square(z_mean) - tf.square(tf.exp(z_log_var)) + 1)
        self.add_loss(kl_loss)
        return reconstructed, z_mean, z_log_var, z


# Y, P, GT, n_epochs, latent_dim_ratio, repeat, name, setting
def run_cluster_latents(X_train, y_train, X_val, y_val, X_test, y_test, 
                        n_epochs, latent_dim_ratio, repeat, name, setting,):

    # Initialization
    latent_dim = int(X_train.shape[1] / latent_dim_ratio)  # Dimension of latent variables
    original_dim = X_train.shape[1]  # Dimension of original datapoints
    intermediate_dim = original_dim

    vae = VariationalAutoEncoder(original_dim=original_dim, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    mse_loss_fn_ = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()
    loss_metric_ = tf.keras.metrics.Mean()

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    spec = str(n_epochs) + "-" + str(latent_dim) + "-" + name + "-" + str(setting) + "-" + str(repeat)

    print("latent dim:", latent_dim)

    step = 0

    for epoch in range(1, n_epochs + 1):
        step += 1
        train_Z = np.array([]).reshape(latent_dim, 0)  # Latent variables (train set)
        train_X_hat = np.array([]).reshape(original_dim, 0)  # reconstructed data points (train set)

        Z_val = np.array([]).reshape(int(latent_dim), 0)  # Latent variables (validation set)
        val_X_hat = np.array([]).reshape(int(original_dim), 0)  # reconstructed data points (validation set)

        # Training the model by iterating over the batches of dataset
        for x_batch_train, _ in train_ds:
            with tf.GradientTape() as tape:
                reconstructed, z_mean, z_log_var, z = vae(x_batch_train)
                # compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KL Divergence regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_metric(loss)

            # if step % 50 == 0:
            #     print('step %s: mean loss train = %s' % (step, loss_metric.result()))

        # Evaluating the performance of the model is done on validation set.
        for x_batch_val, _ in val_ds:
            with tf.GradientTape() as tape:
                reconstructed_, z_mean_, z_log_var_, z_ = vae(x_batch_val)
                Z_val = np.c_[Z_val, z_.numpy().T]
                val_X_hat = np.c_[val_X_hat, reconstructed_.numpy().T]  # concatenating reconstructed data points

                # compute reconstruction loss
                loss_ = mse_loss_fn_(x_batch_val, reconstructed_)
                loss_ += sum(vae.losses)  # Add KL Divergence regularization loss

            grads_ = tape.gradient(loss_, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads_, vae.trainable_weights))

            loss_metric_(loss_)

            # if step % 75 == 0:
            #     print('step %s: mean loss val = %s' % (step, loss_metric_.result()))

        Z_val = Z_val.T

    # K-menas
    kmeans_Z_val = KMeans(n_clusters=len(set(y_val)), n_jobs=-2).fit(Z_val)
    kmeans_val_labels = kmeans_Z_val.labels_

    # Agglomerative (merge)
    agglomerative_val = AgglomerativeClustering(n_clusters=len(set(y_val))).fit(Z_val)
    agg_val_labels = agglomerative_val.labels_

    # Gaussian Mixture Model
    gmm_val = mixture.GaussianMixture(n_components=len(set(y_val), )).fit(Z_val)
    gmm_val_labels = gmm_val.predict(Z_val)

    print("Dev set ARI: ")
    print("K-means:", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=kmeans_val_labels),
          "Agg: ", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=agg_val_labels),
          "GMM: ", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=gmm_val_labels)
          )

    print("Dev set NMI: ")
    print("K-means: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                            labels_pred=kmeans_val_labels,
                                                            average_method='max'),

          "Agg: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                        labels_pred=agg_val_labels,
                                                        average_method='max'),

          "GMM: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                        labels_pred=gmm_val_labels,
                                                        average_method='max'),
          )

    Z_test = np.array([]).reshape(int(latent_dim), 0)  # Latent variables (test set)
    X_test_hat = np.array([]).reshape(int(original_dim), 0)  # reconstructed data points (test set)

    for x_batch_ts, _ in test_ds:
        with tf.GradientTape() as tape:
            reconstructed__, z_mean__, z_log_var__, z__ = vae(x_batch_ts)
            Z_test = np.c_[Z_test, z__.numpy().T]
            X_test_hat = np.c_[X_test_hat, reconstructed__.numpy().T]  # concatenating reconstructed data points

    Z_test = Z_test.T
    X_test_hat = X_test_hat.T

    # K-means
    kmeans_X_test = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(X_test)
    kmeans_X_test_labels = kmeans_X_test.labels_

    kmeans_X_test_hat = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(X_test_hat)
    kmeans_X_test_hat_labels = kmeans_X_test_hat.labels_

    kmeans_Z_test = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(Z_test)
    kmeans_Z_test_labels = kmeans_Z_test.labels_
    kmeans_test_labels = [kmeans_X_test_labels, kmeans_X_test_hat_labels, kmeans_Z_test_labels]

    # Agglomerative (merge)
    agglomerative_X_test = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(X_test)
    agg_X_test_labels = agglomerative_X_test.labels_

    agglomerative_X_test_hat = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(X_test_hat)
    agg_X_test_hat_labels = agglomerative_X_test_hat.labels_

    agglomerative_Z_test = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(Z_test)
    agg_Z_test_labels = agglomerative_Z_test.labels_
    agg_test_labels = [agg_X_test_labels, agg_X_test_hat_labels, agg_Z_test_labels]

    # Gaussian Mixture Model
    gmm_X_test = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(X_test)
    gmm_X_test_labels = gmm_X_test.predict(X_test)

    gmm_X_test_hat = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(X_test_hat)
    gmm_X_test_hat_labels = gmm_X_test_hat.predict(X_test_hat)

    gmm_Z_test = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(Z_test)
    gmm_Z_test_labels = gmm_Z_test.predict(Z_test)
    gmm_test_labels = [gmm_X_test_labels, gmm_X_test_hat_labels, gmm_Z_test_labels]

    return kmeans_test_labels, agg_test_labels, gmm_test_labels, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Path', type=str, default='/home/sshalileh/ml4dc/matrices/',
                        help='Path to load the data sets')

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--Run', type=int, default=0,
                        help='Whether to run the program or to evaluate the results')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z',
                        help='string determining which pre processing method should be applied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    parser.add_argument('--Setting', type=str, default='all')

    parser.add_argument('--Latent_dim_ratio', type=float, default=10.,
                        help='A float denoting the ratio between original data dimension and the latent dimension')

    parser.add_argument('--N_epochs', type=int, default=500,
                        help='An int. denoting the number of epochs')

    args = parser.parse_args()
    path, name, run, with_noise, pp, setting_, latent_dim_ratio, n_epochs = args_parser(args)

    start = time.time()

    if run == 1:

        with open(os.path.join(path, name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        data_name = name.split('(')[0][-2:]
        if with_noise == 1:
            data_name = data_name + "-N"
        type_of_data = name.split('(')[0][-1]
        print("run:", run, name, pp, with_noise, setting_, data_name, type_of_data)

        def apply_aec(data_type, with_noise):  # auto-encode cluster

            # Global initialization
            kmeans_ms = {}  # K-means results
            agg_ms = {}  # Agglomerative results
            gmm_ms = {}  # Gaussian Mixture Model
            GT_ms = {}  # Ground Truth

            if setting_ != 'all':

                for setting, repeats in DATA.items():

                    if str(setting) == setting_:

                        print("setting:", setting, )

                        kmeans_ms[setting] = {}
                        agg_ms[setting] = {}
                        gmm_ms[setting] = {}
                        GT_ms[setting] = {}

                        for repeat, matrices in repeats.items():
                            print("repeat:", repeat)

                            X_tr = DATA[setting][repeat]['X_tr'].astype('float32')
                            X_vl = DATA[setting][repeat]['X_vl'].astype('float32')
                            X_ts = DATA[setting][repeat]['X_ts'].astype('float32')

                            y_tr = DATA[setting][repeat]['y_tr'].astype('float32')
                            y_vl = DATA[setting][repeat]['y_vl'].astype('float32')
                            y_ts = DATA[setting][repeat]['y_ts'].astype('float32')

                            _, _, Xz_tr, _, Xr_tr, _, = ds.preprocess_Y(Yin=X_tr, data_type='Q')
                            _, _, Xz_vl, _, Xr_vl, _, = ds.preprocess_Y(Yin=X_vl, data_type='Q')
                            _, _, Xz_ts, _, Xr_ts, _, = ds.preprocess_Y(Yin=X_ts, data_type='Q')

                            # Different Pre-processing methods
                            if data_type == "NP".lower() and with_noise == 0:
                                print("No Pre-Proc.")
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    X_train=X_tr, y_train=y_tr, X_val=X_vl, y_val=y_vl, X_test=X_ts, y_test=y_ts,
                                    n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=name, setting=setting)

                            elif data_type == "z".lower() and with_noise == 0:
                                print("Z-score")
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    X_train=Xz_tr, y_train=y_tr, X_val=Xz_vl, y_val=y_vl, X_test=Xz_ts, y_test=y_ts,
                                    n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=name, setting=setting)

                            elif data_type == "rng".lower() and with_noise == 0:
                                print("Rng")

                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    X_train=Xr_tr, y_train=y_tr, X_val=Xr_vl, y_val=y_vl, X_test=Xr_ts, y_test=y_ts,
                                    n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=name, setting=setting)

                            kmeans_ms[setting][repeat] = kmeans_labels
                            agg_ms[setting][repeat] = agg_labels
                            gmm_ms[setting][repeat] = gmm_labels
                            GT_ms[setting][repeat] = y_test

                    print("Algorithm is applied on the" + setting_ + "data set!")

            if setting_ == 'all':

                for setting, repeats in DATA.items():

                    print("setting:", setting, )

                    kmeans_ms[setting] = {}
                    agg_ms[setting] = {}
                    gmm_ms[setting] = {}
                    GT_ms[setting] = {}

                    for repeat, matrices in repeats.items():
                        print("repeat:", repeat)

                        X_tr = DATA[setting][repeat]['X_tr'].astype('float32')
                        X_vl = DATA[setting][repeat]['X_vl'].astype('float32')
                        X_ts = DATA[setting][repeat]['X_ts'].astype('float32')

                        y_tr = DATA[setting][repeat]['y_tr'].astype('float32')
                        y_vl = DATA[setting][repeat]['y_vl'].astype('float32')
                        y_ts = DATA[setting][repeat]['y_ts'].astype('float32')

                        _, _, Xz_tr, _, Xr_tr, _, = ds.preprocess_Y(Yin=X_tr, data_type='Q')
                        _, _, Xz_vl, _, Xr_vl, _, = ds.preprocess_Y(Yin=X_vl, data_type='Q')
                        _, _, Xz_ts, _, Xr_ts, _, = ds.preprocess_Y(Yin=X_ts, data_type='Q')

                        # Different Pre-processing methods
                        if data_type == "NP".lower() and with_noise == 0:
                            print("No Pre-Proc.")
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                X_train=X_tr, y_train=y_tr, X_val=X_vl, y_val=y_vl, X_test=X_ts, y_test=y_ts,
                                n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                name=name, setting=setting)

                        elif data_type == "z".lower() and with_noise == 0:
                            print("Z-score")
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                X_train=Xz_tr, y_train=y_tr, X_val=Xz_vl, y_val=y_vl, X_test=Xz_ts, y_test=y_ts,
                                n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                name=name, setting=setting)

                        elif data_type == "rng".lower() and with_noise == 0:
                            print("Rng")

                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                X_train=Xr_tr, y_train=y_tr, X_val=Xr_vl, y_val=y_vl, X_test=Xr_ts, y_test=y_ts,
                                n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                name=name, setting=setting)

                        kmeans_ms[setting][repeat] = kmeans_labels
                        agg_ms[setting][repeat] = agg_labels
                        gmm_ms[setting][repeat] = gmm_labels
                        GT_ms[setting][repeat] = y_test

                print("Algorithm is applied on the entire data set!")

            return kmeans_ms, agg_ms, gmm_ms, GT_ms

        kmeans_ms, agg_ms, gmm_ms, GT_ms = apply_aec(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()
        print("Time:", end - start)

        if with_noise == 1:
            name = name + '-N'

        if setting_ != 'all':

            # Saving K-Means results
            with open(os.path.join('VAE-C_computation', "kmeans_ms_" +
                                                        name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('VAE-C_computation', "agg_ms_" +
                                                        name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('VAE-C_computation', "gmm_ms_" +
                                                        name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            # Saving the corresponding Ground Truth
            with open(os.path.join('VAE-C_computation', "GT_ms_" +
                                                        name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        if setting_ == 'all':

            with open(os.path.join('VAE-C_computation', "kmeans_ms_" +
                                                        name + "-" + pp + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('VAE-C_computation', "agg_ms_" +
                                                        name + "-" + pp + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('VAE-C_computation', "gmm_ms_" +
                                                        name + "-" + pp + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            with open(os.path.join('VAE-C_computation', "GT_ms_" + name + "-" + pp + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        print("Results are saved!")

        for case in range(len(CASES)):

            # res_clt_m_k, re_clt_m_a, re_clt_m_g = adm.evaluation_with_clustering_metrics_several(kmeans_ms=kmeans_ms,
            #                                                                          agg_ms=agg_ms,
            #                                                                          gmm_ms=gmm_ms,
            #                                                                          gt_ms=GT_ms,
            #                                                                          case=-case)
            #
            # for setting, eval_k in res_clt_m_k.items():
            #     eval_a = re_clt_m_a[setting]
            #     eval_g = re_clt_m_g[setting]
            #
            #     print("setting:", setting,
            #           "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
            #           "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
            #           "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
            #           )

            res_clf_m_k, re_clf_m_a, re_clf_m_g = adm.evaluation_with_classification_metric_several(kmeans_ms=kmeans_ms,
                                                                                        agg_ms=agg_ms,
                                                                                        gmm_ms=gmm_ms,
                                                                                        gt_ms=GT_ms,
                                                                                        case=case)

            for setting, eval_k in res_clf_m_k.items():
                eval_a = re_clf_m_a[setting]
                eval_g = re_clf_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_k[4], "%.3f" % eval_k[5],

                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_a[4], "%.3f" % eval_a[5],

                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3],
                      "%.3f" % eval_g[4], "%.3f" % eval_g[5],
                      )

    if run == 0:

        print(" \t", " \t", "name:", name)

        if with_noise == 1:
            name = name + '-N'

        with open(os.path.join('VAE-C_computation', "kmeans_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            kmeans_ms = pickle.load(fp)

        with open(os.path.join('VAE-C_computation', "agg_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            agg_ms = pickle.load(fp)

        with open(os.path.join('VAE-C_computation', "gmm_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            gmm_ms = pickle.load(fp)

        with open(os.path.join('VAE-C_computation', "GT_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            GT_ms = pickle.load(fp)

        for case in range(len(CASES)):

            # res_clt_m_k, re_clt_m_a, re_clt_m_g = adm.evaluation_with_clustering_metrics_several(kmeans_ms=kmeans_ms,
            #                                                                          agg_ms=agg_ms,
            #                                                                          gmm_ms=gmm_ms,
            #                                                                          gt_ms=GT_ms,
            #                                                                          case=case)
            #
            # for setting, eval_k in res_clt_m_k.items():
            #     eval_a = re_clt_m_a[setting]
            #     eval_g = re_clt_m_g[setting]
            #
            #     print("setting:", setting,
            #           "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
            #           "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
            #           "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
            #           )

            res_clf_m_k, re_clf_m_a, re_clf_m_g = adm.evaluation_with_classification_metric_several(
                kmeans_ms=kmeans_ms, agg_ms=agg_ms, gmm_ms=gmm_ms, gt_ms=GT_ms, case=case)

            for setting, eval_k in res_clf_m_k.items():
                eval_a = re_clf_m_a[setting]
                eval_g = re_clf_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_k[4], "%.3f" % eval_k[5],

                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_a[4], "%.3f" % eval_a[5],

                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3],
                      "%.3f" % eval_g[4], "%.3f" % eval_g[5],
                      )

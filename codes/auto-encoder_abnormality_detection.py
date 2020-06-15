"""
This is an implementation of "Abnormality Detection by Auto-Encoder" algorithm for
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
import tempfile
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import ADmetrics as adm
from sklearn import metrics
import data_standardization as ds

BATCH_SIZE = 100
CASES = ['original']
warnings.filterwarnings('ignore')

tf.keras.backend.set_floatx('float32')


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


def flat_ground_truth(ground_truth):
    """
    :param ground_truth: the clusters/communities cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: two flat lists, the first one is the list of labels in an appropriate format
             for applying sklearn metrics. And the second list is the list of lists of
              containing indices of nodes in the corresponding cluster.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        tmp_indices = []
        for vv in range(v):
            labels_true.append(k)
            tmp_indices.append(interval+vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices


class Encoder(tf.keras.layers.Layer):
    def __init__(self, original_dim, latent_dim):
        super(Encoder, self).__init__()
        self.h1 = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu,
                                        input_shape=(original_dim,))
        self.h2 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu,)
        # self.h3 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu,)
        self.h4 = tf.keras.layers.Dense(units=int(original_dim/4), activation=tf.nn.relu,)
        # self.h5 = tf.keras.layers.Dense(units=int(original_dim/8), activation=tf.nn.relu,)
        self.z = tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu)

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        # x = self.h3(x)
        x = self.h4(x)
        # x = self.h5(x)
        z = self.z(x)
        return z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.h1 = tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu,
                                        input_shape=(latent_dim,))
        # self.h2 = tf.keras.layers.Dense(units=int(original_dim/8), activation=tf.nn.relu, )
        self.h3 = tf.keras.layers.Dense(units=int(original_dim/4), activation=tf.nn.relu,)
        # self.h4 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu, )
        self.h5 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu,)
        self.x_hat = tf.keras.layers.Dense(units=original_dim, )

    def call(self, z):
        z = self.h1(z)
        # z = self.h2(z)
        z = self.h3(z)
        # z = self.h4(z)
        z = self.h5(z)
        x = self.x_hat(z)
        return x


class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(original_dim=original_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, original_dim=original_dim)

    def call(self, x):
        coded = self.encoder(x)  # z/latent features/bottle neck
        decoded = self.decoder(coded)  # recmonstructed
        return coded, decoded


def computation(model, original):
    latent_variables = model.encoder(original)
    reconstructed = model.decoder(latent_variables)
    reconstruction_error = tf.losses.mean_squared_error(reconstructed, original)
    return latent_variables, reconstructed, reconstruction_error


def train(computation, model, opt, original):
    with tf.GradientTape() as tape:
        latent_variables, reconstructed, reconstruction_error = computation(model, original)
        gradients = tape.gradient(reconstruction_error, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))


def run_ae(X_train, y_train, X_val, y_val, X_test, y_test,
           n_epochs, latent_dim_ratio, repeat, name, setting):

    X_train = X_train[np.where(y_train == 0), :]
    y_train = y_train[np.where(y_train == 0), :]

    # X_val = X_val[np.where(y_val==0)]
    # y_val = y_val[np.where(y_val==0)]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # Initialization
    latent_dim = int(X_train.shape[1] / latent_dim_ratio)  # Dimension of latent variables
    original_dim = X_train.shape[1]  # Dimension of original datapoints
    spec = str(n_epochs) + "-" + str(latent_dim) + "-" + name + "-" + str(setting) + "-" + str(repeat)

    autoencoder = AutoEncoder(latent_dim=latent_dim, original_dim=original_dim)
    opt = tf.optimizers.Adam(learning_rate=1e-6)

    train_loss_miniBatche = []  # Total loss per Mini-batches (train set)
    train_loss_total_ave = []  # Average loss per each epoch (train set)
    val_loss_miniBatche = []  # Total loss per Mini-batches (validation set)
    val_loss_total_ave = []  # Average loss per each epoch (validation set)
    delta = 1e-4  # The difference between two consecutive validation losses for early stop
    early_stop = False
    early_stop_counter = 0
    patience = 40  # Number of epochs to calculate the differences between consecutive epochs
    
    print("latent dim:", latent_dim)
    
    for epoch in range(1, n_epochs + 1):
        
        print("epoch:", epoch)

        train_Z = np.array([]).reshape(latent_dim, 0)  # Latent variables (train set)
        train_X_hat = np.array([]).reshape(original_dim, 0)  # reconstructed data points (train set)
        train_loss_epoch = []  # Loss values of Mini-batches per each epoch (train set)

        Z_val = np.array([]).reshape(int(latent_dim), 0)  # Latent variables (validation set)
        X_val_hat = np.array([]).reshape(int(original_dim), 0)  # reconstructed data points (validation set)
        val_loss_epoch = []  # Loss values of Mini-batches per each epoch (validation set)

        # Training the model
        for X_tr, _ in train_ds:
            train(computation, autoencoder, opt, X_tr)
            codes, decodes, loss_values = computation(autoencoder, X_tr)
            train_Z = np.c_[train_Z, codes.numpy().T]  # concatenating latent variables
            train_X_hat = np.c_[
                train_X_hat, decodes.numpy().T]  # concatenating reconstructed data points
            train_loss_epoch += loss_values.numpy().tolist()  # appending loss values

        train_loss_total_ave.append(np.mean(np.array(train_loss_epoch)))
        train_loss_miniBatche += [i for i in train_loss_epoch]

        # Evaluating the performance of the model is done on validation set.
        # To stop the training procedure we used early stop condition on validation/dev set.
        for X_vl, _ in val_ds:
            train(computation, autoencoder, opt, X_vl)
            codes_, decodes_, loss_values_ = computation(autoencoder, X_vl)
            Z_val = np.c_[Z_val, codes_.numpy().T]  # concatenating latent variables
            X_val_hat = np.c_[X_val_hat, decodes_.numpy().T]  # concatenating reconstructed data points
            val_loss_epoch += loss_values_.numpy().tolist()  # appending loss values

        val_loss_total_ave.append(np.mean(np.array(val_loss_epoch)))
        val_loss_miniBatche += [i for i in val_loss_epoch]

    # Auto-encoder for abnormality detection
    X_val_hat = X_val_hat.T
    AE_X_val_mse = np.mean(np.power(X_val - X_val_hat, 2), axis=1)
    df_error_X_val = pd.DataFrame({'reconstruction_error': AE_X_val_mse, 'y_test': y_val}, )

    AE_X_val_labels = (df_error_X_val.reconstruction_error > np.mean(X_train)).tolist()
    AE_X_val_labels = [1 if i is True else 0 for i in AE_X_val_labels]

    print("Dev set roc auc: ")
    print("AE Result: ", metrics.roc_auc_score(y_val, AE_X_val_labels, average='weighted'))

    print("Dev set ARI: ")
    print("AE:", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=AE_X_val_labels),)

    print("Dev set NMI: ")
    print("AE: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                       labels_pred=AE_X_val_labels,
                                                       average_method='max')
          )

    with tempfile.TemporaryDirectory() as tmpdirname:
        autoencoder.save_weights(os.path.join(tmpdirname, "AE-" + str(repeat) + spec + ".h5"))
        print("Training finished!")
        autoencoder.load_weights(os.path.join(tmpdirname, "AE-" + str(repeat) + spec + ".h5"))

    Z_test = autoencoder.encoder(X_test).numpy()  # Latent Variables
    X_test_hat = autoencoder.decoder(
        Z_test).numpy()  # reconstructed data points

    # Auto-encoder for abnormality detection
    AE_X_test_mse = np.mean(np.power(X_test - X_test_hat, 2), axis=1)
    df_error_X_test = pd.DataFrame({'reconstruction_error': AE_X_test_mse, 'y_test': y_test}, )

    AE_X_test_labels = (df_error_X_test.reconstruction_error > np.mean(X_train)).tolist()
    AE_X_test_labels = [1 if i is True else 0 for i in AE_X_test_labels]

    return AE_X_test_labels, y_test


if __name__ == '__main__':

    alg_name = os.path.basename(__file__).split(".")[0]

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
            AE_ms = {}  # K-means results
            GT_ms = {}  # Ground Truth

            if setting_ != 'all':

                for setting, repeats in DATA.items():

                    if str(setting) == setting_:

                        print("setting:", setting, )

                        AE_ms[setting] = {}
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

                                AE_X_test_labels, y_test = run_ae(
                                    X_train=X_tr, y_train=y_tr, X_val=X_vl, y_val=y_vl, X_test=X_ts, y_test=y_ts,
                                    n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=name, setting=setting)

                            elif data_type == "z".lower() and with_noise == 0:

                                print("Z-score")

                                AE_X_test_labels, y_test = run_ae(
                                    X_train=Xz_tr, y_train=y_tr, X_val=Xz_vl, y_val=y_vl, X_test=Xz_ts, y_test=y_ts,
                                    n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=name, setting=setting)

                            elif data_type == "rng".lower() and with_noise == 0:

                                print("Rng")

                                AE_X_test_labels, y_test = run_ae(
                                    X_train=Xr_tr, y_train=y_tr, X_val=Xr_vl, y_val=y_vl, X_test=Xr_ts, y_test=y_ts,
                                    n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=name, setting=setting)

                            AE_ms[setting][repeat] = AE_X_test_labels
                            GT_ms[setting][repeat] = y_test

                    print("Algorithm is applied on the" + setting_ + "data set!")

            if setting_ == 'all':

                for setting, repeats in DATA.items():

                    print("setting:", setting, )

                    AE_ms[setting] = {}
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

                            AE_X_test_labels, y_test = run_ae(
                                X_train=X_tr, y_train=y_tr, X_val=X_vl, y_val=y_vl, X_test=X_ts, y_test=y_ts,
                                n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                name=name, setting=setting)

                        elif data_type == "z".lower() and with_noise == 0:

                            print("Z-score")

                            AE_X_test_labels, y_test = run_ae(
                                X_train=Xz_tr, y_train=y_tr, X_val=Xz_vl, y_val=y_vl, X_test=Xz_ts, y_test=y_ts,
                                n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                name=name, setting=setting)

                        elif data_type == "rng".lower() and with_noise == 0:

                            print("Rng")

                            AE_X_test_labels, y_test = run_ae(
                                X_train=Xr_tr, y_train=y_tr, X_val=Xr_vl, y_val=y_vl, X_test=Xr_ts, y_test=y_ts,
                                n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                name=name, setting=setting)

                        AE_ms[setting][repeat] = AE_X_test_labels
                        GT_ms[setting][repeat] = y_test

                print("Algorithm is applied on the entire data set!")

            return AE_ms, GT_ms

        AE_ms, GT_ms = apply_aec(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()
        print("Time:", end - start)

        if with_noise == 1:
            name = name + '-N'

        if setting_ != 'all':

            # Saving AE results
            with open(os.path.join('AE-C_computation', "AE_ms_" +
                                                        name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(AE_ms, fp)

            # Saving the corresponding Ground Truth
            with open(os.path.join('AE-C_computation', "GT_ms_" +
                                                        name + "-" + pp + "-" + setting_ + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        if setting_ == 'all':

            with open(os.path.join('AE-C_computation', "AE_ms_" +
                                                        name + "-" + pp + ".pickle"), 'wb') as fp:
                pickle.dump(AE_ms, fp)

            with open(os.path.join('AE-C_computation', "GT_ms_" + name + "-" + pp + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        print("Results are saved!")

        for case in range(len(CASES)):

            res_clt_m_k = adm.evaluation_with_clustering_metrics(alg_ms=AE_ms, gt_ms=GT_ms, case=-case)

            for setting, eval_k in res_clt_m_k.items():

                print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                      "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      )

            res_clf_m_k = adm.evaluation_with_classification_metric(alg_ms=AE_ms, gt_ms=GT_ms, case=case)

            for setting, eval_k in res_clf_m_k.items():
                print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                      "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_k[4], "%.3f" % eval_k[5],
                      )

    if run == 0:

        print(" \t", " \t", "name:", name)

        if with_noise == 1:
            name = name + '-N'

        with open(os.path.join('AE-C_computation', "AE_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            AE_ms = pickle.load(fp)

        with open(os.path.join('AE-C_computation', "GT_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            GT_ms = pickle.load(fp)

        for case in range(len(CASES)):

            res_clt_m_k = adm.evaluation_with_clustering_metrics(alg_ms=AE_ms, gt_ms=GT_ms, case=case)

            for setting, eval_k in res_clt_m_k.items():

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      )

            res_clf_m_k = adm.evaluation_with_classification_metric(alg_ms=AE_ms, gt_ms=GT_ms, case=case)

            for setting, eval_k in res_clf_m_k.items():
                print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                      "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_k[4], "%.3f" % eval_k[5],
                      )

        adm.plot_curves_of_an_algorithm(alg_ms=AE_ms, gt_ms=GT_ms,
                                        data_name=name, alg_name=alg_name,
                                        case=0, sample_weight=None)

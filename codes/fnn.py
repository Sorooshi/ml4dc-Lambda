"""

This is an implementation of "A Three-Layer Fully Connected Neural Network with dropout" algorithm for
ML4DC  studies on CERN data set.
 These studies are devoted to the task of abnormality detection on CERN's experimental results.
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
import argparse
import ADmetrics as adm
import tensorflow as tf
import data_standardization as ds


BATCH_SIZE = 100
CASES = ['original', 'reconstructed', 'latent']


def args_parser(args):
    path = args.Path
    name = args.Name
    run = args.Run
    pp = args.PreProcessing
    n_epochs = args.N_epochs
    return path, name, run, pp, n_epochs


class ThreeLayerNN(tf.keras.Model):

    def __init__(self, original_dim, name='fnn', **kwargs):
        super(ThreeLayerNN, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.layer_1 = tf.keras.layers.Dense(int(original_dim/2), activation='relu',
                                             input_shape=(original_dim, None))
        self.dropout_1 = tf.keras.layers.Dropout(0.25)
        self.layer_2 = tf.keras.layers.Dense(int(original_dim/4), activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.25)
        self.layer_3 = tf.keras.layers.Dense(int(original_dim/8), activation='relu')
        self.layer_4 = tf.keras.layers.Dense(2, activation='sigmoid')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.dropout_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x


def run_the_algorithm(X_train, y_train, X_val, y_val, X_test, y_test, n_epochs, repeat, ds_name, setting):

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # Initialization
    original_dim = X_train.shape[1]  # Dimension of original datapoints
    spec = 'fnn_clf' + str(n_epochs) + "-" + ds_name + "-" + str(setting) + "-" + str(repeat)

    fnn_clf = ThreeLayerNN(original_dim=original_dim)
    opt = tf.optimizers.Adam(learning_rate=1e-6)
    binary_loss_fn = tf.keras.losses.BinaryCrossentropy()
    binary_loss_fn_ = tf.keras.losses.BinaryCrossentropy()
    loss_metric = tf.keras.metrics.Mean()
    loss_metric_ = tf.keras.metrics.Mean()

    print("original dim:", original_dim)

    step = 0
    for epoch in range(1, n_epochs+1):
        step += 1
        for x_batch_tr, y_tr in train_ds:
            with tf.GradientTape() as tape:
                labels_pred = fnn_clf(x_batch_tr)
                labels_true = tf.reshape(tf.tile(y_tr, [2]), [-1, 2])
                loss = binary_loss_fn(labels_true, labels_pred)

        grads = tape.gradient(loss, fnn_clf.trainable_weights)
        opt.apply_gradients(zip(grads, fnn_clf.trainable_weights))
        loss_metric(loss)

        if step % 50 == 0:
            print('step %s: mean loss train = %s' % (step, loss_metric.result()))

        for x_batch_vl, y_val in val_ds:
            with tf.GradientTape() as tape:
                labels_pred_ = fnn_clf(x_batch_vl)
                labels_true_ = tf.reshape(tf.tile(y_val, [2]), [-1, 2])
                loss_ = binary_loss_fn_(labels_true_, labels_pred_)

        grads_ = tape.gradient(loss_, fnn_clf.trainable_weights)
        opt.apply_gradients(zip(grads_, fnn_clf.trainable_weights))

        loss_metric_(loss_)

        if step % 75 == 0:
            print('step %s: mean loss val = %s' % (step, loss_metric_.result()))

    alg_x_test_probabilities = fnn_clf.predict(X_test)
    alg_x_test_labels = tf.argmax(alg_x_test_probabilities, axis=1)

    return alg_x_test_labels, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Path', type=str, default='/home/sshalileh/ml4dc/matrices/',
                        help='Path to load the data sets')

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--Run', type=int, default=0,
                        help='Whether to run the program or to evaluate the results')

    parser.add_argument('--PreProcessing', type=str, default='NP',
                        help='string determining which pre processing method should be applied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    parser.add_argument('--N_epochs', type=int, default=500,
                        help='An int. denoting the number of epochs')

    args = parser.parse_args()
    path, name, run, pp, n_epochs = args_parser(args)

    start = time.time()

    print("Name of data set:", name, "PreProcessing method:", pp,)

    if run == 1:

        with open(os.path.join(path, name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)


        def apply_the_algorithm(data_type,):  # auto-encode cluster

            # Global initialization
            alg_ms = {}  # the algorithm results
            gt_ms = {}  # Ground Truth

            for setting, repeats in DATA.items():

                print("setting:", setting, )

                alg_ms[setting] = {}
                gt_ms[setting] = {}

                for repeat, matrices in repeats.items():
                    print("repeat:", repeat)

                    X_tr = DATA[setting][repeat]['X_tr'].astype('float32')
                    X_vl = DATA[setting][repeat]['X_vl'].astype('float32')
                    X_ts = DATA[setting][repeat]['X_ts'].astype('float32')

                    y_tr = DATA[setting][repeat]['y_tr'].astype('float32')
                    y_vl = DATA[setting][repeat]['y_vl'].astype('float32')
                    y_ts = DATA[setting][repeat]['y_ts'].astype('float32')

                    y_tr = [1 if i == 1 else -1 for i in y_tr]
                    y_vl = [1 if i == 1 else -1 for i in y_vl]
                    y_ts = [1 if i == 1 else -1 for i in y_ts]

                    _, _, Xz_tr, _, Xr_tr, _, = ds.preprocess_Y(Yin=X_tr, data_type='Q')
                    _, _, Xz_vl, _, Xr_vl, _, = ds.preprocess_Y(Yin=X_vl, data_type='Q')
                    _, _, Xz_ts, _, Xr_ts, _, = ds.preprocess_Y(Yin=X_ts, data_type='Q')

                    # Different Pre-processing methods
                    if data_type == "NP".lower():
                        print("No Pre-Proc.")

                        alg_x_test_labels, y_test = run_the_algorithm(X_train=X_tr, y_train=y_tr,
                                                                      X_val=X_vl, y_val=y_vl,
                                                                      X_test=X_ts, y_test=y_ts,
                                                                      n_epochs=n_epochs,
                                                                      repeat=repeat, ds_name=name,
                                                                      setting=setting
                                                                      )

                    elif data_type == "z".lower():

                        print("Z-score")

                        alg_x_test_labels, y_test = run_the_algorithm(X_train=Xz_tr, y_train=y_tr,
                                                                      X_val=Xz_vl, y_val=y_vl,
                                                                      X_test=Xz_ts, y_test=y_ts,
                                                                      n_epochs=n_epochs,
                                                                      repeat=repeat, ds_name=name,
                                                                      setting=setting
                                                                      )

                    elif data_type == "rng".lower():

                        print("Rng")

                        alg_x_test_labels, y_test = run_the_algorithm(X_train=Xr_tr, y_train=y_tr,
                                                                      X_val=Xr_vl, y_val=y_vl,
                                                                      X_test=Xr_ts, y_test=y_ts,
                                                                      n_epochs=n_epochs,
                                                                      repeat=repeat, ds_name=name,
                                                                      setting=setting)

                    alg_ms[setting][repeat] = alg_x_test_labels
                    gt_ms[setting][repeat] = y_test

                print("Algorithm is applied on the" + str(setting) + "data set!")

            return alg_ms, gt_ms


        alg_ms, gt_ms = apply_the_algorithm(data_type=pp.lower())

        end = time.time()
        print("Time:", end - start)

        # Saving the Classification Results
        with open(os.path.join('FNN_computation', "fnn_ms_" + name + "-" + pp +
                                                  "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
            pickle.dump(alg_ms, fp)

        with open(os.path.join('FNN_computation', "GT_ms_" + name + "-" + pp +
                                                  "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
            pickle.dump(gt_ms, fp)

        res_clt_m_k = adm.evaluation_with_clustering_metrics(alg_ms=alg_ms, gt_ms=gt_ms, case=0)

        for setting, eval_k in res_clt_m_k.items():
            print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                  "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                  )

        res_clf_m_k = adm.evaluation_with_classification_metric(alg_ms=alg_ms, gt_ms=gt_ms, case=0)

        for setting, eval_k in res_clf_m_k.items():
            print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                  "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                  "%.3f" % eval_k[4], "%.3f" % eval_k[5],
                  )

    else:

        # Loading the Classification Results
        with open(os.path.join('FNN_computation', "fnn_ms_" + name + "-" + pp +
                                                  "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            alg_ms = pickle.load(fp)

        with open(os.path.join('FNN_computation', "GT_ms_" + name + "-" + pp +
                                                  "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            gt_ms = pickle.load(fp)

        res_clt_m_k = adm.evaluation_with_clustering_metrics(alg_ms=alg_ms, gt_ms=gt_ms, case=0)

        for setting, eval_k in res_clt_m_k.items():
            print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                  "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                  )

        res_clf_m_k = adm.evaluation_with_classification_metric(alg_ms=alg_ms, gt_ms=gt_ms, case=0)

        for setting, eval_k in res_clf_m_k.items():
            print("setting:", setting, "%.3f" % eval_k[0], "%.3f" % eval_k[1],
                  "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                  "%.3f" % eval_k[4], "%.3f" % eval_k[5],
                  )


"""

This is an implementation of "Isolation Forest" algorithm for ML4DC  studies on CERN data set.
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
import data_standardization as ds
import ADmetrics as adm
from sklearn.ensemble import IsolationForest

CASES = ['original', 'reconstructed', 'latent']


def args_parser(args):
    path = args.Path
    name = args.Name
    pp = args.PreProcessing

    return path, name, pp


def run_the_algorithm(X_train, y_train, X_val, y_val, X_test, y_test,):

    alg_inst = IsolationForest(n_estimators=100, max_samples=2000,
                                       max_features=1, bootstrap=False,
                                       contamination='auto', n_jobs=-2)
    alg_inst.fit(X_train)
    alg_x_train_labels = alg_inst.predict(X_train)  # for future usage if it is needed
    alg_x_val_labels = alg_inst.predict(X_val)  # for future usage if it is needed
    alg_x_test_labels = alg_inst.predict(X_test)

    return alg_x_test_labels, y_test


if __name__ == '__main__':

    alg_name = os.path.basename(__file__).split(".")[0]
    print("1:", alg_name)

    parser = argparse.ArgumentParser()

    parser.add_argument('--Path', type=str, default='/home/sshalileh/ml4dc/matrices/',
                        help='Path to load the data sets')

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--PreProcessing', type=str, default='NP',
                        help='string determining which pre processing method should be applied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    args = parser.parse_args()
    path, name, pp = args_parser(args)

    start = time.time()

    with open(os.path.join(path, name + ".pickle"), 'rb') as fp:
        DATA = pickle.load(fp)

    print("Name of data set:", name, "PreProcessing method:", pp,)

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
                                                                  )

                elif data_type == "z".lower():

                    print("Z-score")

                    alg_x_test_labels, y_test = run_the_algorithm(X_train=Xz_tr, y_train=y_tr,
                                                                  X_val=Xz_vl, y_val=y_vl,
                                                                  X_test=Xz_ts, y_test=y_ts,)

                elif data_type == "rng".lower():

                    print("Rng")

                    alg_x_test_labels, y_test = run_the_algorithm(X_train=Xr_tr, y_train=y_tr,
                                                                  X_val=Xr_vl, y_val=y_vl,
                                                                  X_test=Xr_ts, y_test=y_ts,)

                alg_ms[setting][repeat] = alg_x_test_labels
                gt_ms[setting][repeat] = y_test

            print("Algorithm is applied on the" + str(setting) + "data set!")

        return alg_ms, gt_ms


    alg_ms, gt_ms = apply_the_algorithm(data_type=pp.lower())

    end = time.time()
    print("Time:", end - start)

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

    adm.plot_curves_of_an_algorithm(alg_ms=alg_ms, gt_ms=gt_ms,
                                    data_name=name, alg_name=alg_name,
                                    case=0, sample_weight=None)

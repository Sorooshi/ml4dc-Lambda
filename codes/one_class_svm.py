

import os
import time
import pickle
import argparse
import data_standardization as ds
import ADmetrics as adm
from sklearn.svm import OneClassSVM

CASES = ['original', 'reconstructed', 'latent']

def args_parser(args):
    path = args.Path
    name = args.Name
    pp = args.PreProcessing

    return path, name, pp


def run_the_algorithm(X_train, y_train, X_val, y_val, X_test, y_test,):

    alg_inst = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale', shrinking=True)
    alg_inst.fit(X_train)
    alg_x_train_labels = alg_inst.predict(X_train)  # for future usage if it is needed
    alg_x_val_labels = alg_inst.predict(X_val)  # for future usage if it is needed
    alg_x_test_labels = alg_inst.predict(X_test)

    return alg_x_test_labels, y_test


if __name__ == '__main__':

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


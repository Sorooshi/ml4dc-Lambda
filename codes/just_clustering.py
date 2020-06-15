"""

This is an implementation of "K-Means, Agglomerative, Gaussian Mixture Model"  algorithm for
ML4DC studies on CERN data set.
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
import warnings
import argparse
import tempfile
import numpy as np
from sklearn import metrics
from sklearn import mixture
import data_standardization as ds
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split


BATCH_SIZE = 64
warnings.filterwarnings('ignore')
CASES = ['original', 'reconstructed', 'latent']


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
            tmp_indices.append(interval + vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices


def evaluation_with_clustering_metrics(kmeans_ms, agg_ms, gmm_ms, GT_ms, case):

    recm_k, recm_a, recm_g = {}, {}, {}  # Results Evaluation with Clustering Metrics

    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    print("\t", "  p", "  q", " a/e", "\t",
          "  K-ARI  ", "  K-NMI ", "\t",
          "  A-ARI  ", "  A-NMI ", "\t",
          "  G-ARI  ", "  G-NMI  ",
          )

    print(" \t", " \t", " \t",
          " Ave ", " std  ", " Ave ", "std ",
          " Ave ", " std  ", " Ave ", "std ",
          " Ave ", " std  ", " Ave ", "std"
          )

    for setting, results in kmeans_ms.items():

        ARI_k, NMI_k, ARI_a, NMI_a, ARI_g, NMI_g = [], [], [], [], [], []
        for repeat, result in results.items():
            gt = GT_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]
            ARI_k.append(metrics.adjusted_rand_score(gt, lp_k))
            NMI_k.append(metrics.normalized_mutual_info_score(gt, lp_k, average_method='max'))

            ARI_a.append(metrics.adjusted_rand_score(gt, lp_a))
            NMI_a.append(metrics.normalized_mutual_info_score(gt, lp_a, average_method='max'))

            ARI_g.append(metrics.adjusted_rand_score(gt, lp_g))
            NMI_g.append(metrics.normalized_mutual_info_score(gt, lp_g, average_method='max'))

        ari_ave_k = np.mean(np.asarray(ARI_k), axis=0)
        ari_std_k = np.std(np.asarray(ARI_k), axis=0)
        nmi_ave_k = np.mean(np.asarray(NMI_k), axis=0)
        nmi_std_k = np.std(np.asarray(NMI_k), axis=0)
        recm_k[setting] = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans
        # recm_k = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans

        ari_ave_a = np.mean(np.asarray(ARI_a), axis=0)
        ari_std_a = np.std(np.asarray(ARI_a), axis=0)
        nmi_ave_a = np.mean(np.asarray(NMI_a), axis=0)
        nmi_std_a = np.std(np.asarray(NMI_a), axis=0)
        recm_a[setting] = [ari_ave_a, ari_std_a, nmi_ave_a, nmi_std_a]  # Evaluation Results Clustering Agglomerative
        # recm_a = [ari_ave_a, ari_std_a, nmi_ave_a, nmi_std_a]  # Evaluation Results Clustering Agglomerative

        ari_ave_g = np.mean(np.asarray(ARI_g), axis=0)
        ari_std_g = np.std(np.asarray(ARI_g), axis=0)
        nmi_ave_g = np.mean(np.asarray(NMI_g), axis=0)
        nmi_std_g = np.std(np.asarray(NMI_g), axis=0)
        recm_g[setting] = [ari_ave_g, ari_std_g, nmi_ave_g, nmi_std_g]  # Evaluation Results Clustering GMM
        # recm_g = [ari_ave_g, ari_std_g, nmi_ave_g, nmi_std_g]  # Evaluation Results Clustering GMM

    return recm_k, recm_a, recm_g


def evaluation_with_classification_metric(kmeans_ms, agg_ms, gmm_ms, GT_ms, case):
    recm_k, recm_a, recm_g = {}, {}, {}  # Results Evaluation with Classification Metrics
    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    print(" ")
    print("Results on the" + 'f' + "Variable")
    print("\t", "  p", "  q", " a/e   ", "K-roc_auc_score", "A-roc_auc_score", "G-roc_auc_score", )
    print(" \t", " \t", " \t", "Ave", " std", " Ave", " std", " Ave", " std")

    for setting, results in kmeans_ms.items():

        precision_k, recmall_k, fscore_k, roc_auc_k = [], [], [], []
        precision_a, recmall_a, fscore_a, roc_auc_a = [], [], [], []
        precision_g, recmall_g, fscore_g, roc_auc_g = [], [], [], []

        for repeat, result in results.items():
            gt = GT_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]

            tmp_k = metrics.precision_recall_fscore_support(gt, lp_k, average='weighted')
            roc_auc_k.append(metrics.roc_auc_score(gt, lp_k, average='weighted'))

            tmp_a = metrics.precision_recall_fscore_support(gt, lp_a, average='weighted')
            roc_auc_a.append(metrics.roc_auc_score(gt, lp_a, average='weighted'))

            tmp_g = metrics.precision_recall_fscore_support(gt, lp_g, average='weighted')
            roc_auc_g.append(metrics.roc_auc_score(gt, lp_g, average='weighted'))

            precision_k.append(tmp_k[0])
            recmall_k.append(tmp_k[1])
            fscore_k.append(tmp_k[2])

            precision_a.append(tmp_a[0])
            recmall_a.append(tmp_a[1])
            fscore_a.append(tmp_a[2])

            precision_g.append(tmp_g[0])
            recmall_g.append(tmp_g[1])
            fscore_g.append(tmp_g[2])

        # K-means stats
        precision_ave_k = np.mean(np.asarray(precision_k), axis=0)
        precision_std_k = np.std(np.asarray(precision_k), axis=0)

        recmall_ave_k = np.mean(np.asarray(recmall_k), axis=0)
        recmall_std_k = np.std(np.asarray(recmall_k), axis=0)

        fscore_ave_k = np.mean(np.asarray(fscore_k), axis=0)
        fscore_std_k = np.std(np.asarray(fscore_k), axis=0)

        roc_auc_ave_k = np.mean(np.asarray(roc_auc_k), axis=0)
        roc_auc_std_k = np.std(np.asarray(roc_auc_k), axis=0)
        recm_k[setting] = [roc_auc_ave_k, roc_auc_std_k]

        # Agglomerative stats
        precision_ave_a = np.mean(np.asarray(precision_a), axis=0)
        precision_std_a = np.std(np.asarray(precision_a), axis=0)

        recall_ave_a = np.mean(np.asarray(recmall_a), axis=0)
        recmall_std_a = np.std(np.asarray(recmall_a), axis=0)

        fscore_ave_a = np.mean(np.asarray(fscore_a), axis=0)
        fscore_std_a = np.std(np.asarray(fscore_a), axis=0)

        roc_auc_ave_a = np.mean(np.asarray(roc_auc_a), axis=0)
        roc_auc_std_a = np.std(np.asarray(roc_auc_a), axis=0)
        recm_a[setting] = [roc_auc_ave_a, roc_auc_std_a]

        # GMM stats
        precision_ave_g = np.mean(np.asarray(precision_g), axis=0)
        precision_std_g = np.std(np.asarray(precision_g), axis=0)

        recmall_ave_g = np.mean(np.asarray(recmall_g), axis=0)
        recmall_std_g = np.std(np.asarray(recmall_g), axis=0)

        fscore_ave_g = np.mean(np.asarray(fscore_g), axis=0)
        fscore_std_g = np.std(np.asarray(fscore_g), axis=0)

        roc_auc_ave_g = np.mean(np.asarray(roc_auc_g), axis=0)
        roc_auc_std_g = np.std(np.asarray(roc_auc_g), axis=0)
        recm_g[setting] = [roc_auc_ave_g, roc_auc_std_g]

    return recm_k, recm_a, recm_g


def run_cluster_latents(Y, P, GT, n_epochs, latent_dim_ratio, repeat, name, setting):

    X = np.concatenate((Y, P), axis=1)

    # Initialization
    latent_dim = int(X.shape[1] / latent_dim_ratio)  # Dimension of latent variables
    original_dim = X.shape[1]  # Dimension of original datapoints
    intermediate_dim = original_dim

    n_clusters = len(GT)

    if len(name) == 2:  # because the length of the real-world datasets are all larger than two strings
        y, _ = flat_ground_truth(GT)
    else:
        y = GT

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60,
                                                        random_state=42, shuffle=True)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,
                                                    random_state=42, shuffle=True)

    # K-means
    kmeans_X_test = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(X_test)
    kmeans_X_test_labels = kmeans_X_test.labels_

    kmeans_X_test_hat_labels = kmeans_X_test_labels

    kmeans_Z_test_labels = kmeans_X_test_labels
    kmeans_test_labels = [kmeans_X_test_labels, kmeans_X_test_hat_labels, kmeans_Z_test_labels]

    # Agglomerative (merge)
    agglomerative_X_test = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(X_test)
    agg_X_test_labels = agglomerative_X_test.labels_

    agg_X_test_hat_labels = agglomerative_X_test.labels_

    agg_Z_test_labels = agglomerative_X_test.labels_
    agg_test_labels = [agg_X_test_labels, agg_X_test_hat_labels, agg_Z_test_labels]

    # Gaussian Mixture Model
    # gmm_X_test = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(X_test)
    gmm_X_test_labels = agg_X_test_hat_labels  # gmm_X_test.predict(X_test)

    gmm_X_test_hat_labels = gmm_X_test_labels # gmm_X_test_labels

    gmm_Z_test_labels = gmm_X_test_labels
    gmm_test_labels = [gmm_X_test_labels, gmm_X_test_hat_labels, gmm_Z_test_labels]

    return kmeans_test_labels, agg_test_labels, gmm_test_labels, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Path', type=str, default='/home/soroosh/gps1/NNs4clustering/synthetic_data/',
                        help='Path to load the data sets')

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--Run', type=int, default=0,
                        help='Whether to run the program or to evaluate the results')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z-m',
                        help='string determining which pre processing method should be applied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    parser.add_argument('--Setting', type=str, default='all')

    parser.add_argument('--Latent_dim_ratio', type=float, default=16.,
                        help='A float denoting the ratio between original data dimension and the latent dimension')

    parser.add_argument('--N_epochs', type=int, default=500,
                        help='An int. denoting the number of epochs')

    args = parser.parse_args()
    path, name, run, with_noise, pp, setting_, latent_dim_ratio, n_epochs = args_parser(args)

    start = time.time()

    if run == 1:

        with open(os.path.join(path, name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        data_name = name.split('(')[0]
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
                            GT = matrices['GT']
                            Y = matrices['Y'].astype("float32")
                            P = matrices['P'].astype("float32")
                            Yn = matrices['Yn']
                            if len(Yn) != 0:
                                Yn = Yn.astype('float32')
                            N, V = Y.shape

                            # Quantitative case
                            if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                                _, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='Q')

                                if with_noise == 1:
                                    Yn, _, Ynz, _, Ynrng, _, = ds.preprocess_Y(Yin=Yn, data_type='Q')

                            # Because there is no Yn in the case of categorical features.
                            if type_of_data == 'C':
                                enc = OneHotEncoder(sparse=False, categories='auto')
                                Y_oneHot = enc.fit_transform(Y).astype("float32")  # oneHot encoding

                                # for WITHOUT follow-up rescale Y_oneHot and for WITH follow-up
                                # Y_oneHot should be replaced with Y
                                Y, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y_oneHot, data_type='C')

                            if type_of_data == 'M':
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Y_q, _, Yz_q, _, Yrng_q, _, = ds.preprocess_Y(Yin=Y[:, :Vq], data_type='Q')
                                enc = OneHotEncoder(sparse=False, categories='auto', )
                                Y_oneHot = enc.fit_transform(Y[:, Vq:]).astype("float32")  # oneHot encoding

                                # for WITHOUT follow-up rescale Y_oneHot and for WITH follow-up
                                # Y_oneHot should be replaced with Y
                                Y_c, _, Yz_c, _, Yrng_c, _, = ds.preprocess_Y(Yin=Y_oneHot, data_type='C')

                                Y = np.concatenate([Y[:, :Vq], Y_oneHot], axis=1)
                                Yrng = np.concatenate([Yrng_q, Yrng_c], axis=1)
                                Yz = np.concatenate([Yz_q, Yz_c], axis=1)

                                if with_noise == 1:
                                    Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                    Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                    Vqn = (Vq + Vc)  # the column index of which noise model1 starts

                                    _, _, Ynz_q, _, Ynrng_q, _, = ds.preprocess_Y(Yin=Yn[:, :Vq], data_type='Q')

                                    enc = OneHotEncoder(sparse=False, categories='auto', )
                                    Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn]).astype("float32")  # oneHot encoding
                                    # for WITHOUT follow-up rescale Yn_oneHot and for WITH
                                    # follow-up Yn_oneHot should be replaced with Y
                                    Yn_c, _, Ynz_c, _, Ynrng_c, _, = ds.preprocess_Y(Yin=Yn_oneHot, data_type='C')

                                    Y_ = np.concatenate([Yn[:, :Vq], Yn_c], axis=1)
                                    Yrng = np.concatenate([Ynrng_q, Ynrng_c], axis=1)
                                    Yz = np.concatenate([Ynz_q, Ynz_c], axis=1)

                                    _, _, Ynz_, _, Ynrng_, _, = ds.preprocess_Y(Yin=Yn[:, Vqn:], data_type='Q')
                                    Yn_ = np.concatenate([Y_, Yn[:, Vqn:]], axis=1)
                                    Ynrng = np.concatenate([Yrng, Ynrng_], axis=1)
                                    Ynz = np.concatenate([Yz, Ynz_], axis=1)

                            P, _, _, Pu, _, _, Pm, _, _, Pl, _, _ = ds.preprocess_P(P=P)

                            # Pre-processing - Without Noise
                            if data_type == "NP".lower() and with_noise == 0:
                                print("NP")

                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y, P, GT, n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-u".lower() and with_noise == 0:
                                print("z-u")
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-m".lower() and with_noise == 0:

                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yz, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-l".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-u".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yrng, P=Pu, GT=GT, n_epochs=n_epochs,
                                    latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name,
                                    setting=setting)

                            elif data_type == "rng-m".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yrng, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-l".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            # Pre-processing - With Noise
                            if data_type == "NP".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yn, P=P, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-u".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-m".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynz, P=Pm, GT=GT, n_epochs=n_epochs,
                                    latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=data_name, setting=setting)

                            elif data_type == "z-l".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-u".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynrng, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-m".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynrng, P=Pm, GT=GT, n_epochs=n_epochs,
                                    latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=data_name, setting=setting)

                            elif data_type == "rng-l".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

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
                        GT = matrices['GT']
                        Y = matrices['Y'].astype('float32')
                        P = matrices['P'].astype('float32')
                        Yn = matrices['Yn']
                        if len(Yn) != 0:
                            Yn = Yn.astype('float32')
                        N, V = Y.shape

                        # Quantitative case
                        if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                            _, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='Q')
                            if with_noise == 1:
                                Yn, _, Ynz, _, Ynrng, _, = ds.preprocess_Y(Yin=Yn, data_type='Q')

                        # Because there is no Yn in the case of categorical features.
                        if type_of_data == 'C':
                            enc = OneHotEncoder()  # categories='auto')
                            Y = enc.fit_transform(Y).astype('float32')  # oneHot encoding
                            Y = Y.toarray()
                            # Boris's Theory
                            Y, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='C')

                        if type_of_data == 'M':
                            Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                            Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                            Y_, _, Yz_, _, Yrng_, _, = ds.preprocess_Y(Yin=Y[:, :Vq], data_type='M')
                            enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                            Y_oneHot = enc.fit_transform(Y[:, Vq:]).astype('float32')  # oneHot encoding
                            Y = np.concatenate([Y_oneHot, Y[:, :Vq]], axis=1)
                            Yrng = np.concatenate([Y_oneHot, Yrng_], axis=1)
                            Yz = np.concatenate([Y_oneHot, Yz_], axis=1)

                            if with_noise == 1:
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Vqn = (Vq + Vc)  # the column index of which noise model1 starts

                                _, _, Yz_, _, Yrng_, _, = ds.preprocess_Y(Yin=Yn[:, :Vq], data_type='M')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn]).astype('float32')  # oneHot encoding
                                Y_ = np.concatenate([Yn_oneHot, Yn[:, :Vq]], axis=1)
                                Yrng = np.concatenate([Yn_oneHot, Yrng_], axis=1)
                                Yz = np.concatenate([Yn_oneHot, Yz_], axis=1)

                                _, _, Ynz_, _, Ynrng_, _, = ds.preprocess_Y(Yin=Yn[:, Vqn:], data_type='M')
                                Yn_ = np.concatenate([Y_, Yn[:, Vqn:]], axis=1)
                                Ynrng = np.concatenate([Yrng, Ynrng_], axis=1)
                                Ynz = np.concatenate([Yz, Ynz_], axis=1)

                        P, _, _, Pu, _, _, Pm, _, _, Pl, _, _ = ds.preprocess_P(P=P)

                        # Pre-processing - Without Noise
                        if data_type == "NP".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Y, P=P, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-u".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-m".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yz, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-l".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-u".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yrng, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-m".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yrng, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-l".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        # Pre-processing - With Noise
                        if data_type == "NP".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yn, P=P, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-u".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-m".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynz, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-l".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-u".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynrng, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-m".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynrng, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-l".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

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
            with open(os.path.join('VAE-KC_computation', "kmeans_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('VAE-KC_computation', "agg_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('VAE-KC_computation', "gmm_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            # Saving the corresponding Ground Truth
            with open(os.path.join('VAE-KC_computation', "GT_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        if setting_ == 'all':
            with open(os.path.join('VAE-KC_computation', "kmeans_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('VAE-KC_computation', "agg_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('VAE-KC_computation', "gmm_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            with open(os.path.join('VAE-KC_computation', "GT_ms_" + name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        print("Results are saved!")

        for case in range(len(CASES)):

            res_clt_m_k, re_clt_m_a, re_clt_m_g = evaluation_with_clustering_metrics(kmeans_ms=kmeans_ms,
                                                                                     agg_ms=agg_ms,
                                                                                     gmm_ms=gmm_ms,
                                                                                     GT_ms=GT_ms,
                                                                                     case=-case)

            for setting, eval_k in res_clt_m_k.items():
                eval_a = re_clt_m_a[setting]
                eval_g = re_clt_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
                      )

            # res_clf_m_k, re_clf_m_a, re_clf_m_g = evaluation_with_classification_metric(kmeans_ms=kmeans_ms,
            #                                                                             agg_ms=agg_ms,
            #                                                                             gmm_ms=gmm_ms,
            #                                                                             GT_ms=GT_ms,
            #                                                                             case=case)
            #
            # for setting, eval_k in res_clf_m_k.items():
            #     eval_a = re_clf_m_a[setting]
            #     eval_g = re_clf_m_g[setting]
            #
            #     print("setting:", setting,
            #           "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
            #           "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
            #           "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
            #           )

    if run == 0:

        print(" \t", " \t", "name:", name)

        if with_noise == 1:
            name = name + '-N'

        with open(os.path.join('VAE-KC_computation', "kmeans_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            kmeans_ms = pickle.load(fp)

        with open(os.path.join('VAE-KC_computation', "agg_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            agg_ms = pickle.load(fp)

        with open(os.path.join('VAE-KC_computation', "gmm_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            gmm_ms = pickle.load(fp)

        with open(os.path.join('VAE-KC_computation', "GT_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            GT_ms = pickle.load(fp)

        for case in range(len(CASES)):

            res_clt_m_k, re_clt_m_a, re_clt_m_g = evaluation_with_clustering_metrics(kmeans_ms=kmeans_ms,
                                                                                     agg_ms=agg_ms,
                                                                                     gmm_ms=gmm_ms,
                                                                                     GT_ms=GT_ms,
                                                                                     case=case)

            for setting, eval_k in res_clt_m_k.items():
                eval_a = re_clt_m_a[setting]
                eval_g = re_clt_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
                      )

            # res_clf_m_k, re_clf_m_a, re_clf_m_g = evaluation_with_classification_metric(kmeans_ms=kmeans_ms,
            #                                                                             agg_ms=agg_ms,
            #                                                                             gmm_ms=gmm_ms,
            #                                                                             GT_ms=GT_ms,
            #                                                                             case=case)
            #
            # for setting, eval_k in res_clf_m_k.items():
            #     eval_a = re_clf_m_a[setting]
            #     eval_g = re_clf_m_g[setting]
            #
            #     print("setting:", setting,
            #           "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
            #           "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
            #           "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
            #           )
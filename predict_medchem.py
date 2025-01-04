"""
This module defined functions to predict the following predicted properties:

TBD:

- pKa
- LogD
- LogP
- LogS
"""

import pandas
from tqdm import tqdm
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
import xgboost as xgb
# could consider umap instead?
# http://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html
# https://greglandrum.github.io/rdkit-blog/posts/2023-03-02-clustering-conformers.html
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

np.random.seed(42)

def get_generator_error(mols,pka,ids,label, generator, fingerprint_size):
    """

    :param mols: list, N, of moleclules
    :param pka: list, N, of y values to fit
    :param ids: list, N, of ids
    :param label: fingerprint type
    :param generator: a fingerprint to use
    :param fingerprint_size: number of bits for fingerprint
    :return: list, size K, each element a tuple of errors dataframe  and grid
    """
    grid, _, _, _, _, _, _ = fit(mols, pka, radius=2,
                                 fingerprint_size=fingerprint_size, params=None,
                                 n_jobs=-2, generator=generator, groups=ids)
    df_errors = flatten_errors(grid)
    df_errors["bits"] = fingerprint_size
    df_errors["fp_type"] = label
    errors_grid = [df_errors, grid]
    return errors_grid




def distances(fp_list):
    """

    :param fp_list: list, length N, of fingerprints
    :return: list rrepresenting the first argument to Butina.ClusterData
    (i.e., "DistData" array)
    """
    # see https://github.com/PatWalters/workshop/blob/master/clustering/taylor_butina.ipynb
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        # pylint: disable=no-member
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    return dists


def cluster_ids(fp_list, cutoff=0.35):
    """

    :param fp_list: list, length N, of fingerprints
    :param cutoff: passed to Butina.ClusterData; fingerprints closer than
    this are in the same cluster
    :return: list, length N, of arbitrary cluster IDs
    """
    nfps = len(fp_list)
    mol_clusters = Butina.ClusterData(distances(fp_list), nfps, cutoff,
                                      isDistData=True)
    cluster_id_list = [0] * nfps
    for idx, cluster_list in enumerate(mol_clusters, 1):
        for member in cluster_list:
            cluster_id_list[member] = idx
    return cluster_id_list

def _mol_to_fingerprints(mols,fp_generator):
    return np.array([list(e) for e in mols.transform(fp_generator.GetFingerprint)],
                    dtype=bool)

def _grouped_train_test(X,y,groups,validation_size):
    """

    :param X: length N, x values
    :param y: length N, y values
    :param groups:  length N, group ids for each X
    :param validation_size:  percent to save for test set
    :return: tuple of <X_train, X_test, y_train, y_test, groups_trian, groups_test>
    """
    # the test size is a fractio (e.g. 0.1), so the numbers of splits is 1/test_size (e.g., 1/0.1 = 10);
    # first one is the only one we care about (e.g., 90% train 10% test)
    splitter = GroupKFold(n_splits=int(np.ceil(1 / validation_size)))
    splits = list(splitter.split(X=X, y=y, groups=groups))
    train_idx, test_idx = splits[0]
    # make sure all the indices are accounted for
    assert sorted(list(train_idx) + list(test_idx)) == list(range(len(y)))
    # make sure none of the groups intersect
    assert {groups[i] for i in train_idx} & {groups[i] for i in test_idx} == set()
    X_train, X_test = np.array([X[i] for i in train_idx]), np.array(
        [X[i] for i in test_idx])
    y_train, y_test = np.array([y[i] for i in train_idx]), np.array(
        [y[i] for i in test_idx])
    groups_train, groups_test = np.array(
        [groups[i] for i in train_idx]), np.array(
        [groups[i] for i in test_idx])
    return X_train, X_test, y_train, y_test, groups_train, groups_test

def cross_validate(X,y,groups,params,n_folds,validation_size,n_jobs):
    """

    :param X: x values, length N
    :param y:  y values, length N
    :param groups:
    :param params: param grid to use
    :param n_folds: number of folds for cv
    :param validation_size: fraction (e.g. 0.1) to hold out for testing
    :param n_jobs: number of jobs for CV
    :return: tuple of
        tuple of <fitted grid, X_train, X_test, y_train, y_test, groups_trian, groups_test>
    """
    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42,
                                                            test_size=validation_size)
        # just use standard cross validation, not recommended due to overfit possibility
        cv = n_folds
        groups_train = None
        groups_test = None
    else:
        X_train, X_test, y_train, y_test, groups_train, groups_test  = \
            _grouped_train_test(X,y,groups, validation_size)
        # use 5 fold validation when we fit
        cv = GroupKFold(n_splits=n_folds)
    # Use "hist" for constructing the trees, with early stopping enabled.
    model = xgb.XGBRegressor()
    grid = GridSearchCV(estimator=model, cv=cv,
                        param_grid=params,
                        scoring='r2', n_jobs=n_jobs,
                        verbose=False, return_train_score=True)
    grid.fit(X=X_train, y=y_train, groups=groups_train)
    return grid, X_train, X_test, y_train, y_test, groups_train, groups_test

def fit(mols, pka, radius=2, fingerprint_size=512, n_jobs=None, params=None,
        generator=rdFingerprintGenerator.GetMorganGenerator, groups=None,
        n_folds=10, validation_size=0.1):
    """

    :param mols: see get_generator_error
    :param pka: see get_generator_error
    :param radius: radius of the fingerprint
    :param fingerprint_size: see get_generator_error
    :param n_jobs:  number of jobs; negative N would mean "all except N"
    :param params: param grid
    :param generator: see get_generator_error
    :param groups: see get_generator_error
    :param n_folds:  number of folds for training data
    :param validation_size: fraction to hold out as validation data (not fit!)
    these are *held out* from training and testing
    :return: grid, X_train, X_test, y_train, y_test, groups_train, groups_test
    """
    if params is None:
        params = {'max_depth': [1, 2, 3, 4],
                  'n_estimators': [2, 10, 50, 100, 200]}
    if generator is rdFingerprintGenerator.GetRDKitFPGenerator:
        fp_generator = generator(maxPath=2 * radius, fpSize=fingerprint_size)
    elif generator is rdFingerprintGenerator.GetAtomPairGenerator:
        fp_generator = generator(maxDistance=2 * radius,
                                 fpSize=fingerprint_size)
    elif generator is rdFingerprintGenerator.GetTopologicalTorsionGenerator:
        fp_generator = generator(torsionAtomCount=radius,
                                 fpSize=fingerprint_size)
    else:
        fp_generator = generator(radius=radius, fpSize=fingerprint_size)
    return cross_validate(X=_mol_to_fingerprints(mols,fp_generator), y=pka,
                          groups=groups, params=params, n_folds=n_folds,
                          validation_size=validation_size, n_jobs=n_jobs)


def flatten_errors(grid):
    """

    :param grid: first output of  fit
    :return: dataframe flattening the training and test scores of grid
    """
    df_to_cat = []
    for lab, y, y_error in [["test", "mean_test_score", "std_test_score"],
                            ["train", "mean_train_score", "std_train_score"]]:
        df_y = pandas.concat([pandas.DataFrame(grid.cv_results_["params"]),
                              pandas.DataFrame(grid.cv_results_[y],
                                               columns=["Score"]),
                              pandas.DataFrame(grid.cv_results_[y_error],
                                               columns=["Score error"])],
                             axis=1)
        df_y["Set"] = lab
        df_to_cat.append(df_y)
    df_cat = pandas.concat(df_to_cat)
    return df_cat

def cluster(mols,fpSize,cutoff):
    """

    :param mols: see get_generator_error
    :param fpSize: see get_generator_error
    :param cutoff: see get_generator_error
    :return: list of ids corresponding to molecules
    """
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=fpSize)
    fingerprints = list(mols.transform(fp_generator.GetFingerprint))
    ids = cluster_ids(fingerprints,cutoff=cutoff)
    return ids

def compare_fingerprints(mols,pka,ids):
    """

    :param mols: see get_generator_error
    :param pka: see get_generator_error
    :param ids: see get_generator_error
    :return:
    """
    generators = [
        ['ttgen', rdFingerprintGenerator.GetTopologicalTorsionGenerator],
        ['apgen', rdFingerprintGenerator.GetAtomPairGenerator],
        ["mgngen", rdFingerprintGenerator.GetMorganGenerator],
        ['rdkgen', rdFingerprintGenerator.GetRDKitFPGenerator],
    ]
    kws_error = []
    for label, generator in generators:
        for fingerprint_size in [128, 256,512,1024]:
            kws_error.append({"label": label, "generator": generator,
                              "fingerprint_size": fingerprint_size})

    errors_and_grid_array = []
    for kw in tqdm(kws_error):
        errors_grid = get_generator_error(mols=mols,pka=pka,ids=ids,**kw)
        errors_and_grid_array.append(errors_grid)
    return errors_and_grid_array

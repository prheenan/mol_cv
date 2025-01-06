"""
Module to define loading the medchem data
"""
import os.path

import numpy as np
import pandas
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MolToInchi,MolFromInchi, MolFromSmiles
from tqdm import tqdm
import utilities

def _load_data_files(logd_paths_names,col_y):
    """

    :param logd_paths_names: list, each element a tuple of <file path, smiles col, data col>
    :param col_y: ne name for data column
    :return:
    """
    RDLogger.DisableLog('rdApp.*')
    df_to_cat = []
    for path, col_smiles, col_logD in tqdm(logd_paths_names,
                                           desc=f"Loading {col_y} files"):
        df_tmp = pandas.read_csv(path).dropna(subset=col_smiles)
        df_tmp.rename({col_smiles: "SMILES", col_logD: col_y}, axis="columns",
                      inplace=True)
        # onnly get the parseable smiles
        df_tmp = df_tmp[df_tmp["SMILES"].transform(utilities.smiles_parseable)]
        df_tmp["SMILES"] = [utilities.normalize_smiles(s) for s in
                            df_tmp["SMILES"]]
        df_to_cat.append(df_tmp)
    df_cat = pandas.concat(df_to_cat)[["SMILES", col_y]]
    df_cat["InChI"] = df_cat["SMILES"].\
        transform(lambda x: MolToInchi(utilities.smiles_to_mol(x)))
    df_median = df_cat[["InChI", col_y]].groupby("InChI").median(). \
        reset_index(drop=False)
    df_median["mol"] = df_median["InChI"].transform(MolFromInchi)
    RDLogger.EnableLog('rdApp.*')
    return df_median.dropna(subset="mol",ignore_index=True)

def load_log_p():
    """

    :return: logP dataframe, with columns lke InChI, Mol, LogP
    """
    base_dir = os.path.join(os.path.dirname(__file__),"lib/RTlogD/original_data/")
    logd_paths_names = [
        [os.path.join(base_dir, "logp.csv"), "smiles", "logP"],
    ]
    return  _load_data_files(logd_paths_names,col_y="log_p")


def load_log_d():
    """

    :return: logD dataframe, with columns lke InChI, Mol, LogD
    """
    base_dir = os.path.join(os.path.dirname(__file__),"lib/RTlogD/original_data/")
    logd_paths_names = [
        [os.path.join(base_dir, "Lipo_logD.csv"), "smiles", "exp"],
        [os.path.join(base_dir, "DB29-data.csv"), "smiles", "logD"]
    ]
    return  _load_data_files(logd_paths_names,col_y="log_d")

def load_log_s():
    """

    :return: log_s dataframe
    """
    base_dir = os.path.join(os.path.dirname(__file__),"data/")
    logs_path_names = [
        [os.path.join(base_dir,"AqSolDB_LogS_curated-solubility-dataset.csv"),
         "SMILES", "Solubility"],
    ]
    return _load_data_files(logs_path_names, col_y="log_s")

def load_pk_a():
    """

    :return:  dataframe of valid pKa data.
    """
    pka_path = "lib/Dissociation-Constants/iupac_high-confidence_v2_2.csv"
    pKa = pandas.read_csv(os.path.join(os.path.dirname(__file__),pka_path))
    invalid_temps = ["Neutral molecule unstable", "not_stated", "c", "not",
                     "Not stated", "Not given",
                     "Few details", "not stated", "not_stated "]
    pKa["Temperature ('C)"] = \
        [float(str(t).replace("<", ""))
         if t not in invalid_temps else float("nan") for t in pKa["T"]]
    pKa["pk_a"] = [valid_pka_or_None(val)
                   for val in pKa["pka_value"]]
    pKa["Degrees from 20 ('C)"] = np.abs(pKa["Temperature ('C)"] - 20)
    pKa1_temp = pKa[pKa["pka_type"].isin(["pKa1"]) & pKa["Temperature ('C)"].between(15,25)]
    pKa1_median = pKa1_temp[["InChI", "pk_a"]].groupby("InChI").median().reset_index()
    pKa1_median["mol"] = pKa1_median["InChI"].transform(Chem.MolFromInchi)
    for m in pKa1_median["mol"]:
        utilities.normalize_mol_inplace(m)
    pKa1_median.dropna(subset="mol", inplace=True, ignore_index=True)
    pKa1_median.dropna(subset="pk_a", inplace=True, ignore_index=True)
    return pKa1_median

def valid_pka_or_None(val):
    """

    :param val: pka from the IUPAC dataset
    :return: None if not a valid pKa, otherwise a float representing the pka
    """
    s_val = str(val)
    try:
        return float(val)
    except ValueError:
        if "~" in s_val or "<" in s_val or ">" in s_val or "temp" in s_val or "not_stated" in s_val:
            return None
        else:
            to_range = [" to ", "-"]
            for r in to_range:
                if r in s_val:
                    return np.mean([float(f) for f in s_val.split(r)])
            return None

def load_fda_drugs(limit=None):
    """
    :param limit: number of rows/molecules to limit to
    :return: dataframe of approved drugs
    """
    file_path = os.path.join(os.path.dirname(__file__), "data/fda.csv")
    fda = pandas.read_csv(file_path)[:limit]
    RDLogger.DisableLog('rdApp.*')
    fda["smiles"] = fda["smiles"].transform(utilities.normalize_smiles)
    fda["mol"] = fda["smiles"].transform(MolFromSmiles)
    RDLogger.EnableLog('rdApp.*')
    fda.drop_duplicates("smiles", ignore_index=True, inplace=True)
    fda.dropna(subset="smiles", inplace=True, ignore_index=True)
    fda.dropna(subset="mol", inplace=True, ignore_index=True)
    return fda

def name_to_load_functions():
    """

    :return:  dictionary going from property/column name to load function
    """
    return {
        "log_p": load_log_p,
        "log_d": load_log_d,
        "pk_a": load_pk_a,
        "log_s":load_log_s
    }

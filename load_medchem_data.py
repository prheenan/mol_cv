"""
Module to define loading the medchem data
"""
import os.path

import numpy as np
import pandas
from rdkit import Chem

import utilities


def load_pka():
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
    pKa["pka_value"] = [valid_pka_or_None(val)
                        for val in pKa["pka_value"]]
    pKa["Degrees from 20 ('C)"] = np.abs(pKa["Temperature ('C)"] - 20)
    pKa1_temp = pKa[pKa["pka_type"].isin(["pKa1"]) & pKa["Temperature ('C)"].between(15,25)]
    pKa1_median = pKa1_temp[["InChI", "pka_value"]].groupby(
        "InChI").median().reset_index()
    pKa1_median["mol"] = pKa1_median["InChI"].transform(Chem.MolFromInchi)
    for m in pKa1_median["mol"]:
        utilities.normalize_mol_inplace(m)
    pKa1_median.dropna(subset="mol", inplace=True, ignore_index=True)
    pKa1_median.dropna(subset="pka_value", inplace=True, ignore_index=True)
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

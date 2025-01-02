"""
Unit tests for molecular CV
"""
import unittest
import pandas
import numpy as np
from rdkit.Chem import MolFromSmiles
import mol_cv



class MyTestCase(unittest.TestCase):
    """
    Unit tests
    """
    @classmethod
    def setUpClass(cls):
        """
        set up the common data sets used for testing
        """
        cls.fda = pandas.read_csv("./data/fda.csv")
        cls.fda["smiles"] = cls.fda["smiles"].transform(mol_cv._normalize_smile)
        cls.fda["mol"] = cls.fda["smiles"].transform(MolFromSmiles)
        cls.fda.drop_duplicates("smiles",ignore_index=True,inplace=True)
        cls.fda.dropna(subset="smiles",inplace=True,ignore_index=True)
        cls.fda.dropna(subset="mol",inplace=True,ignore_index=True)
        # get pka data
        pKa = pandas.read_csv("lib/Dissociation-Constants/iupac_high-confidence_v2_2.csv")
        invalid_temps = ["Neutral molecule unstable", "not_stated", "c", "not",
                         "Not stated", "Not given",
                         "Few details", "not stated", "not_stated "]
        pKa["Temperature ('C)"] = \
            [float(str(t).replace("<", "")) if t not in invalid_temps else float("nan")
             for t in pKa["T"]]
        pKa["Degrees from 20 ('C)"] = np.abs(pKa["Temperature ('C)"] - 20)
        _ = pKa[pKa["pka_type"].isin(["pKa1"]) & pKa["Temperature ('C)"].between(15, 25)]

    def test_pKa(self):
        """
        test the pKa code
        """

    def test_lilly(self):
        """
        Test the lily scoring routine
        """
        # read in the fda approved drugs
        _ = mol_cv._mols_to_lilly(mols=MyTestCase.fda["mol"])
        assert True

if __name__ == '__main__':
    unittest.main()

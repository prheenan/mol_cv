"""
Unit tests for molecular CV
"""
import unittest
import pandas
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
        cls.fda = pandas.read_csv("./data/fda.csv").drop_duplicates("smiles").\
            reset_index(drop=True)
        cls.fda["mol"] = cls.fda["smiles"].transform(MolFromSmiles)

    def test_lilly(self):
        """
        Test the lily scoring routine
        """
        # read in the fda approved drugs
        mol_cv._mols_to_lilly(mols=MyTestCase.fda["mol"])


if __name__ == '__main__':
    unittest.main()

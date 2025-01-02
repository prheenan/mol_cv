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
    def __init__(self,*args,**kw):
        """
        initialize class

        :param args: passed to super
        :param kw: passed to super
        """
        super().__init__(*args,**kw)
        self.i_subtest = 0

    @classmethod
    def setUpClass(cls):
        """
        set up the common data sets used for testing
        """
        cls.fda = pandas.read_csv("./data/fda.csv")
        cls.fda["smiles"] = cls.fda["smiles"].transform(mol_cv.normalize_smiles)
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
        mols = MyTestCase.fda["mol"]
        smiles = MyTestCase.fda["smiles"]
        lilly_output = mol_cv._mols_to_lilly(mols=mols)
        smiles_match = [l["SMILES"] == s for l, s in zip(lilly_output, smiles)]
        self.i_subtest = 0
        # make sure all the SMILES match
        with self.subTest(i=self.i_subtest):
            assert len(smiles_match) == len(smiles)
            self.i_subtest += 1
        # spot check a few that I did by hand on 2025-01-02
        mol_demerits_explanation = [ \
            ['CC(C)n1c(/C=C/[C@H](O)C[C@H](O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21',33,'too_many_atoms'],
            ['C[C@@H](CCc1ccc(O)cc1)NCCc1ccc(O)c(O)c1',150,"catechol"],
            ['Cn1cc[nH]c1=S',30,'thiocarbonyl_aromatic']
        ]
        lilly_output_dict = { l["SMILES"]:l for l in lilly_output}
        for smi, demerits,explanation in mol_demerits_explanation:
            with self.subTest(i=self.i_subtest,msg=f"{smi} found"):
                assert smi in lilly_output_dict
            self.i_subtest += 1
            output = lilly_output_dict[smi]
            with self.subTest(i=self.i_subtest,msg=f"{smi} demerits {demerits}"):
                assert output["Demerits"] == demerits, output["Demerits"]
            self.i_subtest += 1
            with self.subTest(i=self.i_subtest,msg=f"{smi} explanation {explanation}"):
                assert output["Explanation"] == explanation, output["Explanation"]
            self.i_subtest += 1

if __name__ == '__main__':
    unittest.main()

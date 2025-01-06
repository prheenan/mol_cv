"""
Unit tests for molecular CV
"""
import unittest
import tempfile
import pandas
import numpy as np
from rdkit.Chem import MolFromSmiles, Descriptors, MolToInchi
from rdkit import RDLogger
import mol_cv
import predict_medchem



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
        RDLogger.DisableLog('rdApp.*')
        cls.fda["smiles"] = cls.fda["smiles"].transform(mol_cv.normalize_smiles)
        cls.fda["mol"] = cls.fda["smiles"].transform(MolFromSmiles)
        RDLogger.EnableLog('rdApp.*')
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

    def test_99_predictors(self):
        """
        test all the predictors
        """
        # make sure we can load and fit every model (just N=1000 to speed up)
        self.i_subtest = 0
        for pred in [predict_medchem.log_p__predictor, predict_medchem.pk_a__predictor,
                     predict_medchem.log_s__predictor, predict_medchem.log_d__predictor]:
            predictor = pred(limit=1000, force=True, verbose=True,
                             cache_model=False)
            with tempfile.NamedTemporaryFile(suffix=".json") as f_tmp:
                predictor.save_model(f_tmp.name)
                post_save = predict_medchem.FittedModel().load_model(f_tmp.name)
                attributes = post_save.attributes_are_list()
                for a in attributes:
                    with self.subTest(i=self.i_subtest):
                        equals = getattr(post_save,a) == getattr(predictor,a)
                        try:
                            assert equals.all()
                        except TypeError:
                            assert equals
                    self.i_subtest += 1
                # check that the estimator properties are the same
                attr_estimator = ["feature_importances_", "intercept_"]
                for a in attr_estimator:
                    with self.subTest(i=self.i_subtest):
                        assert all(getattr(post_save.estimator, a) == \
                                   getattr(predictor.estimator, a))
                self.i_subtest += 1
                # make sure the precictions the saved object makes are the same
                with self.subTest(i=self.i_subtest):
                    all(post_save.predict(post_save.X_test) == \
                        predictor.predict(predictor.X_test))
                self.i_subtest += 1

    def test_lilly(self):
        """
        Test the lily scoring routine

        Note in rare cases (<2% of test set), the Lilly code will return
        an alternate version of the SMILES which

        For example, giving

        NC(N)=NC(=O)c1nc(Cl)c(N)nc1N

        as input to lill will result in it outputting

        NC(=N)NC(=O)C1=NC(Cl)=C(N)N=C1N

        This molecule has a hydrogen moved, but is otherwise the same (the inchi match)
        """
        RDLogger.DisableLog('rdApp.*')
        # read in the fda approved drugs
        smiles = MyTestCase.fda["smiles"]
        lilly_output = mol_cv._smiles_to_lilly(smiles=smiles)
        lilly_output_dict = { l["SMILES"]:l for l in lilly_output}
        self.i_subtest = 0
        with self.subTest(i=self.i_subtest):
            f_match = sum(l["SMILES"] == s for l, s in zip(lilly_output, smiles)) / len(smiles)
            # lilly will sometimes modify the smiles a little
            assert f_match > 0.98
        self.i_subtest += 1
        mol_expect = MyTestCase.fda["mol"]
        mol_found = [MolFromSmiles(l["SMILES"]) for l in lilly_output]
        # make sure the molecular weights all match (even if the SMILES
        # are different -- a good example would be
        with self.subTest(i=self.i_subtest):
            np.testing.assert_allclose([Descriptors.ExactMolWt(m) for m in mol_expect],
                                       [Descriptors.ExactMolWt(m) for m in mol_found])
        self.i_subtest += 1
        # make sure the inchi match (checking the mass is redundant, but I am paranoid)
        with self.subTest(i=self.i_subtest):
            assert [MolToInchi(i) for i in mol_expect] == [MolToInchi(i) for i in mol_found]
        self.i_subtest += 1
        # spot check a few that I did by hand on 2025-01-02
        mol_demerits_explanation = [ \
            ['CC(C)n1c(/C=C/[C@H](O)C[C@H](O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21',33,
             'too_many_atoms'],
            ['C[C@@H](CCc1ccc(O)cc1)NCCc1ccc(O)c(O)c1',150,"catechol"],
            ['Cn1cc[nH]c1=S',30,'thiocarbonyl_aromatic']
        ]
        for smi, demerits,explanation in mol_demerits_explanation:
            with self.subTest(i=self.i_subtest,msg="Found"):
                assert smi in lilly_output_dict
            self.i_subtest += 1
            output = lilly_output_dict[smi]
            with self.subTest(i=self.i_subtest,msg=f"Demerits {demerits}"):
                assert output["Demerits"] == demerits, output["Demerits"]
            self.i_subtest += 1
            with self.subTest(i=self.i_subtest,msg=f"Explanation {explanation}"):
                assert output["Explanation"] == explanation, output["Explanation"]
            self.i_subtest += 1
    RDLogger.EnableLog('rdApp.*')

if __name__ == '__main__':
    unittest.main()

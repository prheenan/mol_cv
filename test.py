"""
Unit tests for molecular CV
"""
import unittest
import tempfile
import logging
import pandas
import numpy as np
from rdkit.Chem import MolFromSmiles, Descriptors, MolToInchi, MolToSmiles, MolToMolBlock
from rdkit import RDLogger
from click.testing import CliRunner
import mol_cv
import predict_medchem
import load_medchem_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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
        cls.fda = load_medchem_data.load_fda_drugs()

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

    def test_00_cns_mpo(self):
        """
        tests the cns mpo calculator works as expected
        """
        self.i_subtest =0
        kw_terms_cns_mpo = [
            [{'log_p':3.7, 'log_d':2.7, 'tpsa':90, 'mw':375, 'hbd':1, 'pk_a':9},
             [0.65, 0.65, 0.89, 1.0, 0.83, 0.5],
             4.5,
             2]
        ]
        for kw_table,expected_terms,cns_mpo,round_n in kw_terms_cns_mpo:
            with self.subTest(self.i_subtest):
                # see table 2 of CNS MPO paper
                val = mol_cv.cns_mpo(**kw_table)
                assert np.round(val,1) == cns_mpo
            self.i_subtest += 1
            # make sure the individual terms are calculated correctly
            with self.subTest(self.i_subtest):
                val = mol_cv.cns_mpo_terms(**kw_table)
                # note the terms are returned in the following order
                # TPSA and MW switches relative to table 2
                # log_p, log_d, mw, tpsa, hbd, pk_a
                assert all(np.round(val,round_n) == expected_terms)
            self.i_subtest += 1

    def test_01_lilly(self):
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

    def test_96_predictor_list(self):
        """
        Test the cli functionality of the predictor list of properties
        """
        runner = CliRunner()
        # make sure just listing the properties works
        with self.subTest(self.i_subtest):
            result = runner.invoke(mol_cv.allowed_properties,
                                   catch_exceptions=False)
            assert result.output.strip().split("\n") == mol_cv.all_properties()
        self.i_subtest += 1

    def test_96_predictor_properties(self):
        """
        Test the cli functionality of the predictor properties
        """
        runner = CliRunner()
        mols = [None,None,MolFromSmiles("CCCC")]
        smiles = [ MolToSmiles(m) if m is not None else None
                   for m in mols]
        inchi = [ MolToInchi(m) if m is not None else None
                  for m in mols]
        molfile = [ MolToMolBlock(m) if m is not None else None
                    for m in mols]
        df = pandas.DataFrame({"SMILES":smiles,
                               "INCHI":inchi,
                               "MOL":molfile})
        with (tempfile.NamedTemporaryFile(suffix=".csv") as f_input,
              tempfile.NamedTemporaryFile(suffix=".csv") as f_output):
            args = ['--input_file',f_input.name,
                    '--output_file',f_output.name]
            df.to_csv(f_input.name,index=False)
            for col in df.columns:
                args_col = args + ["--structure_column",col,"--structure_type",col]
                logger.info( "test_96_predictor_properties:: running properties with: {:s}".\
                             format(" ".join(args_col)))
                with self.subTest(self.i_subtest):
                    result = runner.invoke(mol_cv.properties,args_col,
                                           catch_exceptions=False)
                    assert result.exit_code == 0
                self.i_subtest += 1


    def test_98_predictor_fit(self):
        """
        test that the values work on the fda data set
        """
        predictor_dict = predict_medchem.all_predictors()
        fda = load_medchem_data.load_fda_drugs()
        self.i_subtest = 0
        with self.subTest(i=self.i_subtest, msg="Calculating properties"):
            mol_cv.calculate_properties(fda["mol"],predictor_dict)
        self.i_subtest += 1

if __name__ == '__main__':
    unittest.main()

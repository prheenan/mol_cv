"""
module defining the molecular CV
"""
import tempfile
import warnings
import locale
import os
import re
import sys
import subprocess

from  rdkit.Chem.SaltRemover import SaltRemover
import numpy as np
import pandas
import click
from click import ParamType
# pylint: disable=c-extension-no-member
from rdkit.Chem import (rdMolDescriptors, Descriptors,QED,MolToSmiles,
                        MolFromSmiles,MolFromInchi,MolFromMolBlock)
from rdkit import RDLogger
# for more information on filters see:
# see : https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/FilterCatalog/README
# warning catch are needed to avoid the error given below see
# https://github.com/rdkit/rdkit/issues/4425
with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=RuntimeWarning)
    # pylint: disable=no-member
    from rdkit.Chem import FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
import utilities
from utilities import normalize_smiles
import predict_medchem
import cv_plot


pattern_demerits = re.compile(pattern=r"""
                ^
                (?P<SMILES>.+?) # this is the SMILES
                \s+PRH_(?P<id>\d{12}) # this is the id
                (
                \s+:\sD\( # demerit string must be proceeded by ' : D\('
                (?P<demerits>\d+) # demerit is just the number
                \)
                )?
                (\s+\(?
                (?P<explanations>.+?)  # explanation must be proceeded by whitespace
                :?# optional ending colon which we don't want to capture
                )?
                \)?
                \s*
                $ # must match to the end
                """,flags=re.VERBOSE)


class BoolType(ParamType):
    """
    Defines click boolean stlye argument
    """
    def __init__(self):
        """
        Initialization
        """

    def get_metavar(self, param):
        """

        :param param: name of parametr
        :return:  help string
        """
        return 'Choice([TRUE/True/FALSE/False])'


    def convert(self, value, _, __):
        """

        :param value: value to convert
        :param _: not used
        :param __: not used
        :return: boolean value
        """
        upper = str(value).upper()
        if upper == "TRUE":
            return True
        elif upper == "FALSE":
            return False
        else:
            self.fail(f"Invalid value: {value}. Expected TRUE/FALSE")
            return False


class Alerts:
    """
    Defines convenience functions for going from molecules to structural altert
    """
    def __init__(self):
        """
        Initialization function
        """
        self.filters = {}
        for catalogs, label in [
            [FilterCatalogParams.FilterCatalogs.BRENK, "Brenk"],
            [FilterCatalogParams.FilterCatalogs.NIH, "NIH"],
            [FilterCatalogParams.FilterCatalogs.PAINS, "PAINS"]]:
            params = FilterCatalogParams()
            params.AddCatalog(catalogs)
            # pylint: disable=no-member
            filterer = FilterCatalog.FilterCatalog(params)
            self.filters[label] = filterer

    def _general_filter(self, mol,filter_obj):
        """

        :param mol: rdkit molecule
        :param filter_obj: key into self.filters
        :return: all the descriptions for any filter matches
        """
        return [str(filterMatch.filterMatch)
                for filterMatch in filter_obj.GetFilterMatches(mol)]

    def brenk(self, mol):
        """

        :param mol: rdkit molecule
        :return: list of violations of Brenk filters

        Brenk R et al. Lessons Learnt from Assembling Screening Libraries for
       Drug Discovery for Neglected Diseases.
       ChemMedChem 3 (2008) 435-444. doi:10.1002/cmdc.200700139.
        """
        return self._general_filter(mol,self.filters["Brenk"])

    def nih(self, mol):
        """

        :param mol:
        :return: list of violations of NIH filters

         Reference: Doveston R, et al. A Unified Lead-oriented Synthesis of over Fifty
            Molecular Scaffolds. Org Biomol Chem 13 (2014) 859Ð65.
            doi:10.1039/C4OB02287D.
        Reference: Jadhav A, et al. Quantitative Analyses of Aggregation, Autofluorescence,
                and Reactivity Artifacts in a Screen for Inhibitors of a Thiol Protease.
                J Med Chem 53 (2009) 37Ð51. doi:10.1021/jm901070c.
        """
        return self._general_filter(mol,self.filters["NIH"])

    def pains(self, mol):
        """

        :param mol: rdkit molecule
        :return:list of violations of PAINS molecules

        Reference: Baell JB, Holloway GA. New Substructure Filters for Removal of Pan Assay
           Interference Compounds (PAINS) from Screening Libraries and for Their
           Exclusion in Bioassays.
           J Med Chem 53 (2010) 2719Ð40. doi:10.1021/jm901137j.
        """
        return self._general_filter(mol,self.filters["PAINS"])

#instantiate an alert object to speed up future queries
alert_obj = Alerts()

def loss(v, a, b, aa=-float('inf'), bb=-float('inf')):
    """
    cns-mpo style loss function

    :param v: value
    :param a: between bb and a receives top score
    :param b: rightmost bound; between a and b receives linear interpolation
    from 1 to 0; more than this received 0
    :param aa: leftmost bound; less than this receives 0. Set to -inf to disable
    :param bb: between aa and bb receives linear interpolation from 0 to 1; between
    bb and a receives score of 1
    :return: loss score
    """
    if v < aa:
        return 0
    elif v <= bb:
        return (v - aa) / (bb - aa)
    elif v <= a:
        return 1
    elif v < b:
        return (b - v) / (b - a)
    else:
        return 0


def cns_mpo_terms(log_p, log_d, mw, tpsa, hbd, pk_a):
    """

    :param log_p: cLogP
    :param log_d: cLogD
    :param mw:  Molecular Weight
    :param tpsa: polar surface area
    :param hbd: hydeogen bond donors
    :param pk_a: pKa of most basic center
    :return: list of cns mpo terms, in order of arguments. sum to get cns mpo

    Looking at Fig 1E in https://pubs.acs.org/doi/full/10.1021/acschemneuro.6b00029#
    Central Nervous System Multiparameter Optimization Desirability:
    Application in Drug Discovery Article


    From Figure 1E, we know that at HBD=4, the function should be zero
    From Table 2, we know that at HBD=1, the function should be 0.83

    HBD is a little tricky; unlike its appearance it doesn't actually go monotonically
    from 0 to 4. The points (x,y) are given below per WebPlotDigitizer

    1.0065645514223196, 0.8313725490196078
    2.002188183807439, 0.5019607843137255
    2.9978118161925584, 0.16470588235294104

    Loading this into a python variable a and doing

vals = [float(f) for f in a.replace(",","").strip().split()]
x,y = vals[::2], vals[1::2];  import numpy as np; p = np.polyfit(x,y,deg=1); p

    This yields

    array([-0.33479853,  1.16967608])

    In other words

    T0 = (-0.33479853 * HBD)  + 1.16967608

    Setting this equal for T0=1 gives HBD=0.5 and T0=0 gives 3.5 (within 2%):

    (1-p[1])/p[0] # = 0.51
    (0-p[1])/p[0] # = 3.49

    this also gives T0=0.83 at HBD=1 as expected

     np.polyval(p,1) # = (0.8348775407598936

     So the HBD bounds are 0.5 and 3.5. It is wild to me this isnt in the paper
    """
    return (
        loss(log_p, 3, 5),
        loss(log_d, 2, 4),
        loss(mw, 360, 500),
        loss(tpsa, aa=20, bb=40, a=90, b=120),
        loss(hbd, 0.5, 3.5),
        loss(pk_a, 8, 10)
    )


def cns_mpo(*args, **kw):
    """

    :param args: see cns_mpo_terms
    :param kw: see cns_mpo_terms
    :return: cns_mpo score, 0 to 6
    """
    return sum(cns_mpo_terms(*args, **kw))


def _parse_lilly_line_to_dict(l):
    """

    :param l: line output from lilly medchem rules
    :return: dictionary with keys given as <original output, SMILES, Demerits, explanation>
    """
    matched = re.match(pattern_demerits,l)
    if not matched:
        raise ValueError
    demerits = matched.group("demerits")
    if demerits is not None:
        demerits_int = int(demerits)
    else:
        demerits_int = 0
    assert l.split(" ")[0] == matched.group("SMILES")
    explanation = matched.group("explanations")
    row = { "id":int(matched.group("id")),
            "Original output":l,
            "SMILES":normalize_smiles(matched.group("SMILES")),
            "Demerits": demerits_int,
            "Explanation": explanation}
    return row

def _smiles_to_lilly_lines(smiles):
    """

    :param smiles: smles of interest, list, length N
    :return:  tuple of <lilly lines passing, lilly lines failing>
    """
    lilly_dir = os.path.join(os.path.dirname(__file__),"lib/Lilly-Medchem-Rules/")
    with (tempfile.NamedTemporaryFile(suffix=".smi") as input_f,
          tempfile.TemporaryDirectory() as tmp_dir):
        with open(input_f.name, 'w', encoding="utf8") as fh:
            fh.write("\n".join([f"{s} PRH_{i:012d}" for i,s in enumerate(smiles)]))
        args = ["ruby", "Lilly_Medchem_Rules.rb","-B",
                os.path.join(tmp_dir,"bad"),input_f.name,]
        res = subprocess.check_output(args,cwd=lilly_dir)
        if res is not None:
            default_encoding = sys.stdout.encoding
            if default_encoding is not None:
                enconding = default_encoding
            else:
                enconding = locale.getpreferredencoding()
            good_lines = [g.strip()
                          for g in res.decode(enconding).split("\n")]
        else:
            good_lines = []
        good_lines = [g for g in good_lines if len(g) > 0]
        bad_lines = []
        for smi_file in [os.path.join(tmp_dir,f"bad{i}.smi") for i in [0,1,2,3]]:
            with open(os.path.join(lilly_dir, smi_file), 'r',
                      encoding="utf8") as bad_fh:
                bad_lines.extend([r.strip() for r in bad_fh.readlines()])
    return good_lines, bad_lines

def _smiles_to_lilly_dict(smiles,hard_reject_demerits=100):
    """

    :param smiles: list of smiles of interest, size N
    :param hard_reject_demerits: for "Hard rejected" molecules, use this as default
    demerit value
    :return: list of dictonaries, size N, matching smiles
    """
    good_lines, bad_lines = _smiles_to_lilly_lines(smiles)
    good_dict = [_parse_lilly_line_to_dict(l) for l in good_lines]
    for g in good_dict:
        g["Status"] = "Pass"
    bad_dict = [_parse_lilly_line_to_dict(l) for l in bad_lines]
    for b in bad_dict:
        if b["Demerits"] == 0:
            # not given demerits but hard rejected
            # for things that are rejected without demerits, set to default
            b["Status"] = "Hard reject"
            b["Demerits"] = hard_reject_demerits
        else:
            b["Status"] = "Reject"
    all_dicts_ordered = sorted((good_dict + bad_dict), key=lambda x: x["id"])
    assert len(all_dicts_ordered) == len(smiles)
    assert len(set(d["id"] for d in all_dicts_ordered)) == len(all_dicts_ordered)
    return all_dicts_ordered

def _smiles_to_lilly(smiles,**kw):
    """

    :param smiles: list of smiles to pass to lilly
    :param **kw: passedto _smiles_to_lilly_dict
    :return: list of dictionaries, size N, one per molecule
    """
    return _smiles_to_lilly_dict(smiles,**kw)

def mol_cv(mols,**kw):
    """
    Convenience funtion to load all the predictors and calculate all molecular
    properties

    :param mols: list N of molecules
    :param kw: passed to calculate_properties
    :return: calculated properties of molecules
    """
    # instantiate the predictors
    predictor_dict = { k:v() for k,v in predict_medchem.all_predictors().items()}
    return calculate_properties(mols,predictor_dict,**kw)


def all_properties():
    """

    :return: list of all available properties
    """
    ret_naive = list(_name_to_funcs().keys()) + list(predict_medchem.all_predictors().keys()) + \
                ["Total alert count","Lipinski violations","cns_mpo",
                 "Lilly status", "Lilly demerits", "Lilly explanation"]
    # convert the Brent filters etc
    ret_all = []
    for k in ret_naive:
        if k in alert_obj.filters:
            ret_all.append(f"{k} alert count")
            ret_all.append(f"{k} explanation")
        else:
            ret_all.append(k)
    return ret_all

def _calculate_val(mol,func):
    """

    :param mol: molecule
    :param func:  function
    :return: nan is mol is None, otherwise func(mol)
    """
    if mol is None:
        return np.nan
    else:
        return func(mol)

def _row_to_lipinski(row):
    """

    :param row: row with standard outputs defined
    :return:  number of lipinski violations
    """
    return sum(((row["log_p"] > 5),
                (row["Molecular weight"] > 500),
                (row["H-bond donors"] > 5),
                (row["H-bond acceptors"] > 5)))

def _row_to_cns_mpo(row):
    """

    :param row: row with standard outputs defined
    :return: cns mpo
    """
    return cns_mpo(log_p=row["log_p"], log_d=row["log_d"],
                   mw=row["Molecular weight"],
                   tpsa=row["Topological polar surface area"],
                   hbd=row["H-bond donors"],
                   pk_a=row["pk_a"])

def calculate_properties(mols,predictor_dict,limit_to=None,
                         hard_reject_demerits=100):
    """

    :param mols: list N of rdkit Molecule objects
    :param predictor_dict: output of predict_medchem.all_predictors()
    :param limit_to_these: set of string to limit to
    :param hard_reject_demerits: numebr of demerits to give to molecules that
    the Lilly code 'hard rejects'
    :return:
    """
    if limit_to is None:
        # use them all
        limit_to = set(all_properties())
    else:
        limit_to = set(limit_to)
    extra = limit_to - set(all_properties())
    assert extra == set() , f"Didn't understand these properties: {extra}"
    rows = []
    for mol in mols:
        row = {}
        for k, func in _name_to_funcs().items():
            needed = (f"{k} alert count" in limit_to) or (f"{k} explanation" in limit_to)
            if (k in alert_obj.filters) and needed:
                val = _calculate_val(mol, func)
                props =  [ [f"{k} alert count",len(val) if str(val) != "nan" else np.nan],
                           [f"{k} explanation", ",".join(val) if str(val) != "nan" else np.nan]]
                for lab,prop_val in props:
                    # check if we want this property
                    if lab not in limit_to:
                        continue
                    row[lab] = prop_val
            else:
                if k not in limit_to:
                    continue
                val = _calculate_val(mol, func)
                row[k] = val
        if "Total alert count" in limit_to:
            row['Total alert count'] = sum( (row[f"{k} alert count"]
                                             for k in alert_obj.filters
                                             if f"{k} alert count" in limit_to))
        rows.append(row)
    # useful to know which are not null
    i_mols_not_null = [[i, m] for i, m in enumerate(mols) if m is not None]
    i_not_null = [i[0] for i in i_mols_not_null]
    mols_not_null = [i[1] for i in i_mols_not_null]
    predictions_wanted = len(limit_to & predictor_dict.keys()) > 0
    if (len(mols_not_null) == 0) and predictions_wanted:
        # then we can't predict anything (no molecules), but we do want to
        prop_to_idx_value = { k:{} for k in predictor_dict if k in limit_to}
    else:
        # then we  want one or more of the predicted properties and have mols
        # actually call the predictor dicts and create them if they are needed
        # this step is avoided elsewhere since the predictors are expensive
        predictor_dict_instantiated = {k: v() if not isinstance(v,predict_medchem.FittedModel) else v
                                       for k, v in predictor_dict.items()
                                       if k in limit_to}
        # we
        # get all of the predicted properties at once.
        # if we have mols that are none, deal with them gracefully
        prop_to_idx_value = {}
        for prop, pred in predictor_dict_instantiated.items():
            if prop not in limit_to:
                continue
            prop_to_idx_value[prop] = \
                dict(zip(i_not_null,pred.predict_mols(mols_not_null)))
    if predictions_wanted:
        # then at least one prediction wanted; loop through and set
        for i, _ in enumerate(rows):
            for prop_name,dict_v in prop_to_idx_value.items():
                if i not in dict_v:
                    rows[i][prop_name] = np.nan
                else:
                    rows[i][prop_name] = dict_v[i]
    for row in rows:
        if "Lipinski violations" in limit_to:
            # see https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five
            row["Lipinski violations"] = _row_to_lipinski(row)
        if "cns_mpo" in limit_to:
            row["cns_mpo"] = _row_to_cns_mpo(row)
    # lilly is called once for all smiles to speed up
    if "Lilly status" in limit_to or "Lilly demerits" in limit_to or "Lilly explanation" in limit_to:
        # convert to smiles
        smiles = [MolToSmiles(m) for m in mols_not_null]
        if len(mols_not_null) > 0:
            lilly_values = _smiles_to_lilly(smiles,hard_reject_demerits=hard_reject_demerits)
            idx_to_lilly_values =  dict(zip(i_not_null, lilly_values))
        else:
            idx_to_lilly_values = {}
        for i,_ in enumerate(rows):
            for lilly_prop in ["Status", "Demerits", "Explanation"]:
                prop_final = f"Lilly {lilly_prop.lower()}"
                if prop_final in limit_to:
                    if i in idx_to_lilly_values:
                        rows[i][prop_final] = idx_to_lilly_values[i][lilly_prop]
                    else:
                        rows[i][prop_final] = np.nan
    return rows

def _name_to_funcs():
    """
    returns dictionary of property to function giving property

    Still need
    log P
    Lipinski violations
    pKa
    Chemical formula
    Isotope formula
    Dot-disconnected formula
    Composition
    Isotope composition
    """
    return {
         "H-bond donors":rdMolDescriptors.CalcNumHBD,
         "H-bond acceptors":rdMolDescriptors.CalcNumHBA,
         "Exact mass": Descriptors.ExactMolWt,
         "Molecular weight":Descriptors.MolWt,
         "Aromatic rings":rdMolDescriptors.CalcNumAromaticRings,
         "Heavy atoms": rdMolDescriptors.CalcNumHeavyAtoms,
         "Topological polar surface area":rdMolDescriptors.CalcTPSA,
         "Rotatable bonds":rdMolDescriptors.CalcNumRotatableBonds,
         "Chemical formula": rdMolDescriptors.CalcMolFormula,
         "Fraction of carbons SP3 hybridized":rdMolDescriptors.CalcFractionCSP3,
         "QED":QED.qed,
         "Brenk":alert_obj.brenk,
         "NIH":alert_obj.nih,
         "PAINS":alert_obj.pains
    }




def align_and_plot(mols,predictor_dict=None,**kw):
    """

    :param mols: list, length N of molecules
    :param predictor_dict: see output of predict_medchem.all_predictors()
    :param kw: see plot_mol_with_properties
    :return:
    """
    if predictor_dict is None:
        predictor_dict = predict_medchem.all_predictors()
    rows = calculate_properties(mols, predictor_dict)
    return cv_plot.plot_mol_with_properties(mols=mols,rows=rows,**kw)


def _safe_structure_convert_or_None(f_structure,val):
    """

    :param f_structure: convert from value to structure
    :param val:  to convert to structure
    :return: Mol object, or none if conversion failed (type error)
    """
    try:
        return f_structure(val)
    except TypeError:
        return None

@click.group()
def cli():
    """
    defines the click command line group
    """

def _radar_helper(input_file,structure_column,structure_type,
                  normalize_molecules,output_file,**kw):
    """

    :param input_file: input file
    :param structure_column:
    :param structure_type:
    :param normalize_molecules:
    :param output_file: where to output
    :param kw:
    :return:
    """
    if output_file is None:
        output_file = input_file + ".gif"
    mols = _read_and_normalize(input_file,structure_column,structure_type,
                               normalize_molecules)
    return align_and_plot(mols, output_file=output_file,**kw)

@cli.command()
@click.option('--input_file', required=True,
              type=click.Path(exists=True,dir_okay=False),
              help="Name of input file (must be csv)")
@click.option("--structure_column",required=False,type=str,default="SMILES",
              help="name of the structure column in the file")
@click.option("--structure_type",
              type=click.Choice(["MOL","SMILES","INCHI"]),required=False,
              default="SMILES",help="How to read the structure column")
@click.option("--output_file",required=False,type=click.Path(dir_okay=False),
              default=None,help="where to output the file")
@click.option("--normalize_molecules",required=False,type=BoolType(),
              default="FALSE",help="Whether to normalize molecule prior to fitting")
@click.option("--dpi",required=False,type=float,default=300,
              help="dpi of plot")
@click.option("--w_pad",required=False,type=float,default=-5,
              help="width padding")
@click.option("--duration",required=False,type=int,default=2000,
              help="duration of frame padding")
@click.option("--image_height",required=False,type=int,default=None,
              help="mol image height in pixels; generally should not need to set")
@click.option("--image_width",required=False,type=int,default=None,
              help="mol image width in pixels; generally should not need to set")
def radar_plot(**kw):
    """
    make a radar plot
    :param kw: see _radar_helper
    """
    _radar_helper(**kw)


@cli.command()
def allowed_properties():
    """

    :return: print list of allowed properties
    """
    print("\n".join(all_properties()))


def _read_and_normalize(input_file,structure_column,structure_type,
                        normalize_molecules,desalt=False):
    """
    Convenience function for reading moleules from a file

    :param input_file: name of file
    :param structure_column:  name of sturcutre column
    :param structure_type:  one of MOL, SMILES, INCHI
    :param normalize_molecules:  if truem normalize all the molecules
    :param desalt: if true, desalt
    :return:  list of molecules from file
    """
    df = pandas.read_csv(input_file)
    structures = df[structure_column]
    structure_dict = {"MOL":MolFromMolBlock,
                      "SMILES":MolFromSmiles,
                      "INCHI":MolFromInchi}
    structure_f = structure_dict[structure_type]
    RDLogger.DisableLog('rdApp.*')
    mols = [ _safe_structure_convert_or_None(structure_f,s) for s in structures]
    RDLogger.EnableLog('rdApp.*')
    if normalize_molecules:
        for m in mols:
            utilities.normalize_mol_inplace(m)
    if desalt:
        # remove the salt
        remover = SaltRemover()
        m_tmp = []
        for m in mols:
            if m is None:
                res = None
            else:
                try:
                    res = remover.StripMol(m,dontRemoveEverything=True)
                except (ValueError, TypeError):
                    # if can't remove salt
                    res = m
            m_tmp.append(res)
        mols = m_tmp
    return mols

def _properties_helper(input_file,structure_column,structure_type,output_file,
                       limit_to,normalize_molecules,desalt,hard_reject_demerits=100):
    """

    :param input_file: input file csv
    :param structure_column: whattype of columne to use, MOL, SMILES, or InChi
    :param structure_type: see properties
    :param output_file: see properties
    :param limit_to: see properties
    :param desalt: if true, desalt
    :param normalize_molecules: if true, normalize molecules
    :param hard_reject_demerits: number of demerits to give molecules that Lilly
    hard rejects
    :return: nothing
    """
    if output_file is None:
        output_file = input_file + "__mol_cv.csv"
    if limit_to is None:
        limit_list = all_properties()
    else:
        limit_list = [s.strip() for s in limit_to.split(",")]
    mols = _read_and_normalize(input_file,structure_column,structure_type,
                               normalize_molecules,desalt)
    df = pandas.DataFrame(mol_cv(mols,limit_to=limit_list,
                                 hard_reject_demerits=hard_reject_demerits))
    df.to_csv(output_file,index=False)


@cli.command()
@click.option('--input_file', required=True,
              type=click.Path(exists=True,dir_okay=False),
              help="Name of input file (must be csv)")
@click.option("--structure_column",required=False,type=str,
              default="SMILES",
              help="name of the structure column in the file")
@click.option("--structure_type",
              type=click.Choice(["MOL","SMILES","INCHI"]),required=False,
              default="SMILES",help="How to read the structure column")
@click.option("--output_file",required=False,type=click.Path(dir_okay=False),
              default=None,help="where to output the file")
@click.option("--normalize_molecules",required=False,type=BoolType(),
              default="FALSE",help="Whether to normalize molecule prior to fitting")
@click.option('--desalt',required=False,type=BoolType(),
              default="FALSE",help="Whether to desalt prior to calculating everything")
@click.option("--limit_to",required=False,type=str,default=None,
              help="CSV of list of allowed properties (see allowed-properties command). Only calculate these")
@click.option("--hard_reject_demerits",required=False,type=float,default=100,
              help="Demerits to give molecules which lilly rules 'hard reject'")
def properties(**kw):
    """

    :param kw: see properties_helper
    :return: see properties_helper
    """
    _properties_helper(**kw)

if __name__ == '__main__':
    cli()

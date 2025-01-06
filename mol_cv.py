"""
module defining the molecular CV
"""
import tempfile
import os
import re
import sys
import subprocess
# pylint: disable=c-extension-no-member
from rdkit.Chem import (rdMolDescriptors, Descriptors,QED)
from utilities import normalize_smiles

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
        output = subprocess.check_output(args,cwd=lilly_dir).decode(sys.stdout.encoding)
        good_lines = [g.strip() for g in output.split("\n")]
        good_lines = [g for g in good_lines if len(g) > 0]
        bad_lines = []
        for smi_file in [os.path.join(tmp_dir,f"bad{i}.smi") for i in [0,1,2,3]]:
            with open(os.path.join(lilly_dir, smi_file), 'r',
                      encoding="utf8") as bad_fh:
                bad_lines.extend([r.strip() for r in bad_fh.readlines()])
    return good_lines, bad_lines

def _smiles_to_lilly_dict(smiles,default_demerits=100):
    """

    :param smiles: list of smiles of interest, size N
    :param default_demerits: for "Hard rejected" molecules, use this as default
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
            b["Demerits"] = default_demerits
        else:
            b["Status"] = "Reject"
    all_dicts_ordered = sorted((good_dict + bad_dict), key=lambda x: x["id"])
    assert len(all_dicts_ordered) == len(smiles)
    assert len(set(d["id"] for d in all_dicts_ordered)) == len(all_dicts_ordered)
    return all_dicts_ordered

def _smiles_to_lilly(smiles):
    """

    :param smiles: list of smiles to pass to lilly
    :return: list of dictionaries, size N, one per molecule
    """
    return _smiles_to_lilly_dict(smiles)

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
         "Heavy atom count": rdMolDescriptors.CalcNumHeavyAtoms,
         "Topological polar surface area":rdMolDescriptors.CalcTPSA,
         "Rotatable bonds":rdMolDescriptors.CalcNumRotatableBonds,
         "Chemical formula": rdMolDescriptors.CalcMolFormula,
         "QED":QED.qed,
    }

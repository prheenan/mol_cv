"""
module defining the molecular CV
"""
import tempfile
import os
import re
import sys
import subprocess
# pylint: disable=c-extension-no-member
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MolToSmiles

pattern_demerits = re.compile(pattern=r"""
                ^
                (?P<SMILES>.+?) # this is the SMILES
                (
                \s+:\sD\( # demerit string must be proceeded by ' : D\('
                (?P<demerits>\d+) # demerit is just the number
                \)
                )?
                (\s+\(?
                (?P<explanations>.+?)  # explanation must be proceeded by whitespace
                )?
                \)?
                \s*
                $ # must match to the end
                """,flags=re.VERBOSE)

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
    row = { "Original output":l,
            "SMILES":matched.group("SMILES"),
            "Demerits": demerits_int,
            "Explanation": explanation}
    return row

def _smiles_to_lilly_lines(smiles):
    """

    :param smiles: smles of interest
    :return:  tuple of <lilly lines passing, lilly lines failing>
    """
    lilly_dir = os.path.join(os.path.dirname(__file__),"lib/Lilly-Medchem-Rules/")
    with (tempfile.NamedTemporaryFile(suffix=".smi") as input_f,
          tempfile.TemporaryDirectory() as tmp_dir):
        with open(input_f.name, 'w', encoding="utf8") as fh:
            fh.write("\n".join(smiles))
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
    dict_combined =  { list_v["SMILES"]:list_v for list_v in (good_dict + bad_dict)}
    return [dict_combined[s] if s in dict_combined else {} for s in smiles]

def _mols_to_lilly(mols):
    """

    :param mols: list of rdkit molecule objects, size N
    :return: list of dictionaries, size N, one per molecule
    """
    smiles = [MolToSmiles(m) for m in mols]
    dict_v = _smiles_to_lilly_dict(smiles)
    return dict_v

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
         "Exact mass": rdMolDescriptors.ExactMolWt,
         "Heavy atom count": rdMolDescriptors.CalcNumHeavyAtoms,
         "Topological polar surface area":rdMolDescriptors.CalcTPSA,
         "Rotatable bonds":rdMolDescriptors.CalcNumRotatableBonds,
         "Chemical formula": rdMolDescriptors.CalcMolFormula
    }

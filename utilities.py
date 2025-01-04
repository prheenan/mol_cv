# pylint: disable=c-extension-no-member
from rdkit.Chem import (CanonSmiles,MolToSmiles, MolFromSmiles)
from rdkit.Chem.rdmolops import Kekulize

def normalize_mol_inplace(mol):
    """

    :param mol: molecule
    :return: nothing, normalized inplace and returns
    """
    Kekulize(mol, clearAromaticFlags=True)
    return mol

def normalize_smiles(smile):
    """
    Kekulize form especially important for Lilly, see quote/URL below
    https://github.com/IanAWatson/Lilly-Medchem-Rules/issues/3#issuecomment-329043633
    "Generally you will be better off using Kekule forms in files,
    and just compute aromaticity in programs ... We do not recommend using aromatic smiles."

    note that kekulize is in place

    :param smile: smile to use
    """
    mol = MolFromSmiles(CanonSmiles(smile))
    normalize_mol_inplace(mol)
    return CanonSmiles(MolToSmiles(mol))

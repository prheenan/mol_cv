# Introduction

Generate 'molecular CVs' of common chemical properties

## Notes

- For Lilly MedChem rules, the Lilly code rejects some atoms without providing a demerit number.
  - My code gives these all demerits of `default_demerits` and sets their status to `Hard reject`.
  - Molecules with demerit scores are given a status of `Reject`
  - Molecules which pass are given a status of `Pass`

## Data sets used

- logD and LogP from [RTlogD](https://github.com/WangYitian123/RTlogD)
  - > Wang Y, Xiong J, Xiao F, Zhang W, Cheng K, Rao J, Niu B, Tong X, Qu N, Zhang R, Wang D, Chen K, Li X, Zheng M. LogD7.4 prediction enhanced by transferring knowledge from chromatographic retention time, microscopic pKa and logP. J Cheminform. 2023 Sep 5;15(1):76. doi: 10.1186/s13321-023-00754-4. PMID: 37670374; PMCID: PMC10478446.
  - Note this is based on the following (TBD):
    - DB29-data
    - T-data
    - Lipo
    - pKA (TBD??)
- pKa from [Dissociation-Constants](https://github.com/IUPAC/Dissociation-Constants)
  - > Zheng, Jonathan W. and Lafontant-Joseph, Olivier. (2024) IUPAC Digitized pKa Dataset, v2.2. Copyright © 2024 International Union of Pure and Applied Chemistry (IUPAC), The dataset is reproduced by permission of IUPAC and is licensed under a CC BY-NC 4.0. Access at https://doi.org/10.5281/zenodo.7236453.
- [Lilly-Medchem-Rules](https://github.com/IanAWatson/Lilly-Medchem-Rules)
  - > "Rules for Identifying Potentially Reactive or Promiscuous Compounds" by Robert F. Bruns and Ian W. Watson, J. Med. Chem. 2012, 55, 9763--9772
- FDA approved drugs from [Zinc20](https://zinc20.docking.org/substances/subsets/fda/)
  - > Irwin, Tang, Young, Dandarchuluun, Wong, Khurelbaatar, Moroz, Mayfield, Sayle, J. Chem. Inf. Model 2020, in press. https://pubs.acs.org/doi/10.1021/acs.jcim.0c00675. You may also wish to cite our previous papers: Sterling and Irwin, J. Chem. Inf. Model, 2015 http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559. Irwin, Sterling, Mysinger, Bolstad and Coleman, J. Chem. Inf. Model, 2012 DOI: 10.1021/ci3001277 or Irwin and Shoichet, J. Chem. Inf. Model. 2005;45(1):177-82 PDF, DOI.
- LogS from [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1)
  - > Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Sci Data 6, 143 (2019). https://doi.org/10.1038/s41597-019-0151-1

TBD add licensing

## Data sets considered

- cLogD from [logd64](https://github.com/nanxstats/logd74)
  - >    Wang, J-B., D-S. Cao, M-F. Zhu, Y-H. Yun, N. Xiao, Y-Z. Liang (2015). In silico evaluation of logD7.4 and comparison with other prediction methods. Journal of Chemometrics, 29(7), 389-398.


# TODO:

- Properties:

```
log P
H-bond donors
H-bond acceptors
Lipinski violations
pKa
Exact mass
Heavy atom count
Topological polar surface area
Rotatable bonds
Chemical formula
Isotope formula
Dot-disconnected formula
Composition
Isotope composition
```

Also

```
CNS MPO
Lilly MedChem Rules?
LogS?
```

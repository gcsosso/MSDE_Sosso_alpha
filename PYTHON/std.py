import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh,eigh
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP,MolMR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import scipy as sp
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,normalize
from sklearn.model_selection import RepeatedKFold
import collections

# Flatten these nasty nested lists...
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def t3DMOL(mole):
    return AllChem.EmbedMolecule(mole,useRandomCoords=True)

def t3DMOLES(moles):
    vectt3DMOL=np.vectorize(t3DMOL)
    return vectt3DMOL(moles)

def UFF(mole):
    return AllChem.UFFOptimizeMolecule(mole,maxIters=1000)

def UFFSS(moles):
    vecUFF=np.vectorize(UFF)
    return vecUFF(moles)

# Function to get some standard descriptors via RDKit
def STDD(molecules):
    std_descriptors=[]
    i=0
    for m in molecules:
       print("Dealing with molecule n. ",i," of ",len(molecules))
       i=i+1
   
       desc=[]

       # 2D
       desc.append(Chem.GraphDescriptors.BalabanJ(m))        # Chem. Phys. Lett. 89:399-404 (1982)
       desc.append(Chem.GraphDescriptors.BertzCT(m))         # J. Am. Chem. Soc. 103:3599-601 (1981)
       desc.append(Chem.GraphDescriptors.HallKierAlpha(m))   # Rev. Comput. Chem. 2:367-422 (1991)
       desc.append(Chem.Crippen.MolLogP(m))                  # Wildman and Crippen JCICS 39:868-73 (1999)
       desc.append(Chem.Crippen.MolMR(m))                    # Wildman and Crippen JCICS 39:868-73 (1999)
       desc.append(Chem.Descriptors.ExactMolWt(m))
       desc.append(Chem.Descriptors.FpDensityMorgan1(m))
       desc.append(Chem.Descriptors.MaxPartialCharge(m))
       desc.append(Chem.Descriptors.NumRadicalElectrons(m))
       desc.append(Chem.Descriptors.NumValenceElectrons(m))
       desc.append(Chem.Lipinski.FractionCSP3(m))
       desc.append(Chem.Lipinski.HeavyAtomCount(m))
       desc.append(Chem.Lipinski.NumHAcceptors(m))
       desc.append(Chem.Lipinski.NumAromaticRings(m))
       desc.append(Chem.Lipinski.NHOHCount(m))
       desc.append(Chem.Lipinski.NOCount(m))
       desc.append(Chem.Lipinski.NumAliphaticCarbocycles(m))
       desc.append(Chem.Lipinski.NumAliphaticHeterocycles(m))
       desc.append(Chem.Lipinski.NumAliphaticRings(m))
       desc.append(Chem.Lipinski.NumAromaticCarbocycles(m))
       desc.append(Chem.Lipinski.NumAromaticHeterocycles(m))
       desc.append(Chem.Lipinski.NumHDonors(m))
       desc.append(Chem.Lipinski.NumHeteroatoms(m))
       desc.append(Chem.Lipinski.NumRotatableBonds(m))
       desc.append(Chem.Lipinski.NumSaturatedCarbocycles(m))
       desc.append(Chem.Lipinski.NumSaturatedHeterocycles(m))
       desc.append(Chem.Lipinski.NumSaturatedRings(m))
       desc.append(Chem.Lipinski.RingCount(m))
       desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
       desc.append(Chem.rdMolDescriptors.CalcTPSA(m))        # J. Med. Chem. 43:3714-7, (2000)
       desc.append(Chem.MolSurf.LabuteASA(m))                # J. Mol. Graph. Mod. 18:464-77 (2000)

       # 3D
       desc.append(Chem.rdMolDescriptors.CalcAUTOCORR2D(m))
       desc.append(Chem.rdMolDescriptors.CalcAUTOCORR3D(m))
       desc.append(Chem.rdMolDescriptors.CalcAsphericity(m))
       desc.append(Chem.rdMolDescriptors.CalcChi0n(m))
       desc.append(Chem.rdMolDescriptors.CalcChi0v(m))
       desc.append(Chem.rdMolDescriptors.CalcChi1n(m))
       desc.append(Chem.rdMolDescriptors.CalcChi1v(m))
       desc.append(Chem.rdMolDescriptors.CalcChi2n(m))
       desc.append(Chem.rdMolDescriptors.CalcChi2v(m))
       desc.append(Chem.rdMolDescriptors.CalcChi3v(m))
       desc.append(Chem.rdMolDescriptors.CalcChi4n(m))
       desc.append(Chem.rdMolDescriptors.CalcChi4v(m))
       desc.append(Chem.rdMolDescriptors.CalcCrippenDescriptors(m))
       desc.append(Chem.rdMolDescriptors.CalcEccentricity(m))
       desc.append(Chem.rdMolDescriptors.CalcExactMolWt(m))
       desc.append(Chem.rdMolDescriptors.CalcFractionCSP3(m))
       #desc.append(Chem.rdMolDescriptors.CalcGETAWAY(m))
       desc.append(Chem.rdMolDescriptors.CalcHallKierAlpha(m))
       desc.append(Chem.rdMolDescriptors.CalcInertialShapeFactor(m))
       desc.append(Chem.rdMolDescriptors.CalcKappa1(m))
       desc.append(Chem.rdMolDescriptors.CalcKappa2(m))
       desc.append(Chem.rdMolDescriptors.CalcKappa3(m))
       desc.append(Chem.rdMolDescriptors.CalcLabuteASA(m))
       desc.append(Chem.rdMolDescriptors.CalcMORSE(m))
       desc.append(Chem.rdMolDescriptors.CalcNPR1(m))
       desc.append(Chem.rdMolDescriptors.CalcNPR2(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAliphaticRings(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAmideBonds(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumAromaticRings(m))
       #desc.append(Chem.rdMolDescriptors.CalcNumAtomStereoCenters(m))
       desc.append(Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m))
       desc.append(Chem.rdMolDescriptors.CalcNumHBA(m))
       desc.append(Chem.rdMolDescriptors.CalcNumHBD(m))
       desc.append(Chem.rdMolDescriptors.CalcNumHeteroatoms(m))
       desc.append(Chem.rdMolDescriptors.CalcNumHeterocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(m))
       desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(m))
       desc.append(Chem.rdMolDescriptors.CalcNumRings(m))
       desc.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(m))
       desc.append(Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(m))
       desc.append(Chem.rdMolDescriptors.CalcNumSaturatedRings(m))
       desc.append(Chem.rdMolDescriptors.CalcNumSpiroAtoms(m))
       desc.append(Chem.rdMolDescriptors.CalcPBF(m))
       desc.append(Chem.rdMolDescriptors.CalcPMI1(m))
       desc.append(Chem.rdMolDescriptors.CalcPMI2(m))
       desc.append(Chem.rdMolDescriptors.CalcPMI3(m))
       desc.append(Chem.rdMolDescriptors.CalcRDF(m))
       desc.append(Chem.rdMolDescriptors.CalcRadiusOfGyration(m))
       desc.append(Chem.rdMolDescriptors.CalcSpherocityIndex(m))
       desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
       desc.append(Chem.rdMolDescriptors.CalcWHIM(m))
       
       desc=flatten(desc)
       std_descriptors.append(desc)

    return std_descriptors


def STD_ONE(m):
    std_descriptors=[]
    i=0

    desc=[]
 
    # 2D
    desc.append(Chem.GraphDescriptors.BalabanJ(m))        # Chem. Phys. Lett. 89:399-404 (1982)
    desc.append(Chem.GraphDescriptors.BertzCT(m))         # J. Am. Chem. Soc. 103:3599-601 (1981)
    desc.append(Chem.GraphDescriptors.HallKierAlpha(m))   # Rev. Comput. Chem. 2:367-422 (1991)
    desc.append(Chem.Crippen.MolLogP(m))                  # Wildman and Crippen JCICS 39:868-73 (1999)
    desc.append(Chem.Crippen.MolMR(m))                    # Wildman and Crippen JCICS 39:868-73 (1999)
    desc.append(Chem.Descriptors.ExactMolWt(m))
    desc.append(Chem.Descriptors.FpDensityMorgan1(m))
    desc.append(Chem.Descriptors.MaxPartialCharge(m))
    desc.append(Chem.Descriptors.NumRadicalElectrons(m))
    desc.append(Chem.Descriptors.NumValenceElectrons(m))
    desc.append(Chem.Lipinski.FractionCSP3(m))
    desc.append(Chem.Lipinski.HeavyAtomCount(m))
    desc.append(Chem.Lipinski.NumHAcceptors(m))
    desc.append(Chem.Lipinski.NumAromaticRings(m))
    desc.append(Chem.Lipinski.NHOHCount(m))
    desc.append(Chem.Lipinski.NOCount(m))
    desc.append(Chem.Lipinski.NumAliphaticCarbocycles(m))
    desc.append(Chem.Lipinski.NumAliphaticHeterocycles(m))
    desc.append(Chem.Lipinski.NumAliphaticRings(m))
    desc.append(Chem.Lipinski.NumAromaticCarbocycles(m))
    desc.append(Chem.Lipinski.NumAromaticHeterocycles(m))
    desc.append(Chem.Lipinski.NumHDonors(m))
    desc.append(Chem.Lipinski.NumHeteroatoms(m))
    desc.append(Chem.Lipinski.NumRotatableBonds(m))
    desc.append(Chem.Lipinski.NumSaturatedCarbocycles(m))
    desc.append(Chem.Lipinski.NumSaturatedHeterocycles(m))
    desc.append(Chem.Lipinski.NumSaturatedRings(m))
    desc.append(Chem.Lipinski.RingCount(m))
    desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
    desc.append(Chem.rdMolDescriptors.CalcTPSA(m))        # J. Med. Chem. 43:3714-7, (2000)
    desc.append(Chem.MolSurf.LabuteASA(m))                # J. Mol. Graph. Mod. 18:464-77 (2000)

    # 3D
    desc.append(Chem.rdMolDescriptors.CalcAUTOCORR2D(m))
    desc.append(Chem.rdMolDescriptors.CalcAUTOCORR3D(m))
    desc.append(Chem.rdMolDescriptors.CalcAsphericity(m))
    desc.append(Chem.rdMolDescriptors.CalcChi0n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi0v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi1n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi1v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi2n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi2v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi3v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi4n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi4v(m))

    desc.append(Chem.rdMolDescriptors.CalcCrippenDescriptors(m))
    desc.append(Chem.rdMolDescriptors.CalcEccentricity(m))
    desc.append(Chem.rdMolDescriptors.CalcExactMolWt(m))
    desc.append(Chem.rdMolDescriptors.CalcFractionCSP3(m))
    #desc.append(Chem.rdMolDescriptors.CalcGETAWAY(m))
    desc.append(Chem.rdMolDescriptors.CalcHallKierAlpha(m))
    desc.append(Chem.rdMolDescriptors.CalcInertialShapeFactor(m))
    desc.append(Chem.rdMolDescriptors.CalcKappa1(m))
    desc.append(Chem.rdMolDescriptors.CalcKappa2(m))
    desc.append(Chem.rdMolDescriptors.CalcKappa3(m))
    desc.append(Chem.rdMolDescriptors.CalcLabuteASA(m))
    desc.append(Chem.rdMolDescriptors.CalcMORSE(m))
    desc.append(Chem.rdMolDescriptors.CalcNPR1(m))
    desc.append(Chem.rdMolDescriptors.CalcNPR2(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAmideBonds(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticRings(m))
    #desc.append(Chem.rdMolDescriptors.CalcNumAtomStereoCenters(m))

    desc.append(Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHBA(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHBD(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHeteroatoms(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(m))
    desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(m))
    desc.append(Chem.rdMolDescriptors.CalcNumRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSpiroAtoms(m))
    desc.append(Chem.rdMolDescriptors.CalcPBF(m))
    desc.append(Chem.rdMolDescriptors.CalcPMI1(m))
    desc.append(Chem.rdMolDescriptors.CalcPMI2(m))
    desc.append(Chem.rdMolDescriptors.CalcPMI3(m))
    desc.append(Chem.rdMolDescriptors.CalcRDF(m))
    desc.append(Chem.rdMolDescriptors.CalcRadiusOfGyration(m))
    desc.append(Chem.rdMolDescriptors.CalcSpherocityIndex(m))
    desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
    desc.append(Chem.rdMolDescriptors.CalcWHIM(m))


    desc=flatten(desc)
    std_descriptors.append(desc)

    return std_descriptors



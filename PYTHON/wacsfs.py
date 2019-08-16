import pandas as pd
import numpy as np
from numpy import inf
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from mendeleev import element
import h5py
import collections
from scipy.spatial import distance
from scipy.spatial import distance_matrix

m = Chem.MolFromSmiles('CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)')
class SymmetryVec:
    def __init__(self, smile, Nrad, Nang, rcAng, rcRad):
        self.smile = smile
        # self.dist = h5py.File(distPath,'r')
        # self.ang = h5py.File(angPath,'r')
        # self.distMat = self.dist[smile][:]
        # self.angMat = self.ang[smile][:]
        self.Nrad = Nrad
        self.Nang = Nang
        self.rcRad = rcRad
        self.rcAng = rcAng
        self.coords = self.xyz_dict()
        self.distMat = distance_matrix(self.coords,self.coords)
        self.muListRad = self.muList(rcRad, Nrad)
        self.muListAng = self.muList(rcAng, Nang)
        self.etaRad = 1/(2 * np.square((rcRad - 1.5)/(Nrad -1)))
        self.etaAng = 1/(2 * np.square((rcAng - 1.5)/(Nang -1)))
        self.atomList = self.atomList()
        self.weights = [element(atom).atomic_number for atom in self.atomList ]
        self.radVector = self.wACSF_Rad()
        self.angArray = self.makeAngleArray()
        self.angVector =self.wACSF_Ang()

    def unit_vector(self, vector):
        np.seterr(divide='ignore')
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

    def atomList(self):
        m = Chem.MolFromSmiles(self.smile) # The molecule from smiles
        mH = Chem.AddHs(m)
        numAtoms = mH.GetNumAtoms()
        atomList = []
        for i in range(numAtoms):
            atomList.append(mH.GetAtomWithIdx(i).GetSymbol())
        return atomList

    def muList(self, rc, N):
        mu = [0.5]
        for i in range(N-1):
            mu.append(mu[i] + (rc - 1.5)/(N -1))
        return mu

    def xyz_dict(self):
        list = []
        m = Chem.MolFromSmiles(self.smile) # The molecule from smiles
        mH = Chem.AddHs(m)# Adding Hydrogen
        n = mH.GetNumAtoms()
        AllChem.EmbedMolecule(mH) # Initial coordinates of a molecule
        AllChem.UFFOptimizeMolecule(mH) # Optimise molecule?

        for i in range(n) :# For each atom
            coords = []
            pos = mH.GetConformer().GetAtomPosition(i) # Get position coordinates
            coords.append(pos.x)
            coords.append(pos.y)
            coords.append(pos.z)
            list.append(coords)
        return np.array(list)

    def wACSF_Rad(self):
        list = []
        for mu in self.muListRad:
            fvals = 0.5 * (np.cos((self.distMat * np.pi)/self.rcRad) + 1)
            mask = np.where((self.distMat < self.rcRad) & (self.distMat > 0), 1, 0 )
            vals = np.exp(-self.etaRad * (self.distMat-mu)**2)
            unmasked_wACSF = fvals * (vals.T * self.weights).T
            masked_wACSF = unmasked_wACSF * mask
            final = np.sum(masked_wACSF, axis = 0)
            list.append(final)
        return (np.array(list).T)

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

    def makeAngleArray(self):
        coords = self.coords
        n = len(coords)
        array = np.zeros([n,n,n])
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ij = coords[j] - coords[i]
                    ik = coords[k] - coords[i]
                    cosTheta = self.angle_between(ij,ik)
                    # theta = arccos(clip(cosTheta, -1, 1))
                    array[i,j,k] = cosTheta
        return np.nan_to_num(array)

    def wACSF_Ang(self, zeta = 1): # get distMat, atomList from distanceMatrix function. i is int for atom in atomList. angleArray from makeAngleArray function.
        lam = [1, -1]
        list = []
        for l in lam:
            for mu in self.muListAng:
                expMats = np.exp(-self.etaAng*(self.distMat-mu)**2)
                exp3d = np.einsum('ij,ik,jk->ijk',expMats,expMats,expMats)
                fMats = 0.5*(np.cos((self.distMat * np.pi)/self.rcAng)+1)
                f3d = np.einsum('ij,ik,jk->ijk',fMats,fMats,fMats)
                maskMats = np.where((self.distMat < self.rcAng) & (self.distMat > 0), 1, 0 )
                mask3d = np.einsum('ij,ik,jk->ijk',maskMats,maskMats,maskMats)
                angArrayMod = (1 + (self.angArray * l))**zeta
                weights3d = np.einsum('i,j->ij', self.weights,self.weights)
                finWeighted = np.einsum('ijk,ijk,ijk->ijk',exp3d,f3d,angArrayMod)
                finMasked = np.einsum('ijk,ijk->ijk', finWeighted, mask3d)
                wACSFvec = np.einsum('jk,ijk->i', weights3d, finMasked)
                list.append(wACSFvec)
        return np.array(list).T

class histToDf:
    def __init__(self,radVector, angVector, Nrad, Nang, bins):
        self.angArray = angVector
        self.radArray = radVector
        self.Nrad = Nrad
        self.Nang = Nang * 2
        self.bins = bins
        self.minRad, self.maxRad, self.minAng, self.maxAng = self.minMax()

    def minMax(self):
        minRad, maxRad = np.amin(self.radArray), np.amax(self.radArray)
        minAng, maxAng = np.amin(self.angArray), np.amax(self.angArray)
        return minRad, maxRad, minAng, maxAng

    def binningList(self):
        list = []
        for i in range(self.Nrad):
            data = np.histogram(self.radArray[:,i],bins = self.bins,range =(self.minRad,self.maxRad))[0]
            list.extend(data)
        for i in range(self.Nang):
            data = np.histogram(self.angArray[:,i] ,bins = self.bins,range =(self.minAng,self.maxAng))[0]
            list.extend(data)
        return np.array(list)

def histWACSF(smilesList, Nrad, Nang, rcRad, rcAng, bins):
    numCols = (2*Nang +Nrad) * bins
    array = np.empty((0,(numCols)))
    for smile in smilesList:
        data = SymmetryVec(smile, Nrad, Nang, rcAng, rcRad)
        ang = data.angVector
        rad = data.radVector
        temp = histToDf(rad, ang, Nrad, Nang, bins)
        row = temp.binningList().reshape(1,numCols)
        array = np.append(array, row, axis=0)
    return(array)

def makeDF(dataFrameEmpty, histWACSF, Nrad, Nang, bins):
    for i in range(Nrad):
        for j in range(bins):
            dataFrameEmpty['wACSF_Rad{}_{}'.format(i,j)]=0
    for i in range(Nang * 2):
        for j in range(bins):
            dataFrameEmpty['wACSF_Ang{}_{}'.format(i,j)]=0
    dataFrameEmpty.loc[:,'wACSF_Rad0_0':] = histWACSF
    return dataFrameEmpty

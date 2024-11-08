import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import math
import pickle

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski, QED

_fscores = None
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                                2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def compute_properties(smi):
    '''
    '''
    mol = Chem.MolFromSmiles(smi)

    sa = calculateScore(mol)
    qed = round(QED.weights_max(mol), 2)
    mw = Descriptors.ExactMolWt(mol)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(Descriptors.TPSA(mol), 2)
    nhd = Lipinski.NumHDonors(mol)
    nha = Lipinski.NumHAcceptors(mol)

    return smi, sa, qed, mw, logp, tpsa, nhd, nha


if __name__ == '__main__':
    # test here the functions
    smiles_test = ['O=C1C=C/C(=[NH+]\\[O-])N1c1cccc(Cn2ccc(F)c2)c1CF', 'O=c1ccn(CCn2nccc2Nc2ccn(CF)c2)cc1CCO', 'OCc1ccc(COc2ccccc2-n2cc(F)cc2Cl)cc1CO', 'O=c1ccn(Cn2ccc(F)c2-c2cccs2)cc1N1C=C2C=CC=C[C@@H]21', 'O=c1cc(Oc2cccc(O)c2)cc2ccc(CO)cc2c1', 'O=c1ccc(Cn2c[nH+]cc2Cn2c[nH+]cc2CO)ccc1F', 'O=C1OCCN1Nc1cn(-n2nc(CF)cc2CF)ccc1=O', 'O=c1cnn(Cn2c[nH+]cc2Cc2ccn(CF)n2)cc1CCF', 'O=c1cnc(Cn2cc[nH+]c2Cn2cc[nH+]c2)ccc1CF', 'O=Cc1ccccc1-c1ccc(-n2cncc2CO)c(F)c1', 'O=c1ccn(Cc2cccc(SO)n2)c(-c2ccccc2)c1', 'CCc1cc(C=C2CCN(C(=O)C[n+]3ccn(CC)n3)CC2)ccc1CCF', 'O=c1cnc(Nn2c[nH+]cc2-c2ccccc2F)ccc1O', 'O=c1cnc(Nc2ccccc2)cc(-n2cccc2CO)c1F', 'O=c1ccn(-c2cccc(CF)c2)nc1-c1cncnc1F', 'O=c1ccc(Cn2cc(F)cc2-c2ccccc2)cc2ccc(O)cc12', 'O=c1ccc(Cn2cccc2F)ccc1-c1ccn(CO)c1F', 'O=c1ccc(Nc2cc(F)cc[nH+]2)c2ccc(CO)ccc1-2', 'O=c1ncc(Cn2cc[nH+]c2-c2ccn(CF)c2)ccc1CF', 'O=C1N(CCO)Oc2ccc(F)cc2N1c1ccc(CF)cc1CO', 'O=c1ccn(Cc2ccoc2-c2ccc(F)cc2F)cc1CF', 'Cc1cccn1C[C@H](O)CN[C@H]1CCN(c2ccc(O)c(F)c2)O1', 'O=c1ccn(Cc2ccccc2N2OCCc3c[nH+]ccc32)cc1F', 'N#Cc1ccc(/[NH+]=C2/C=C(O)c3ccccc3C2=O)cc1CF', 'O=Cc1ccccc1-c1cn(NN2Cc3ccccc3O2)cn1', 'O=c1ccc(-n2cc(C(F)F)cc2F)cc2c1CCC[NH2+]2', 'O=c1cc(-c2cccc(O)c2F)cc2[nH+]c(-c3ccc(O)cc3)cc-2o1', 'OCc1cc(F)ccc1-n1ccn(-c2cccc(O)c2)ccs1', 'O=c1ccn(-c2cccc(CF)c2F)c(=O)n1[C@H]1CN1', '[O-]c1cc(O)ccc1C1=C(O)CO[N+](c2ccc(CF)cc2)=C1', 'O=c1cc2cc(OCc3ccc(O)cc3)cc-2oc(-c2ccc(O)cn2)c1', '[O-][NH2+]CCc1sc(CN2CCc3ccccc32)cc1CO', 'O=c1ccn(Cn2cccc2-n2ccn2Cc2ccon2)cc1N1CC[C@H]1O', 'OCc1cc(F)c(-c2cccc(N3CCN(O)CC3)c2)s1', 'NC(=O)c1ccc(N2CC[NH2+][C@@H]2CN2CCNc3ccccc32)cc1', 'O=c1cnc(-c2ccn(CO)c2)ccc1-c1ccccc1F', 'O=c1ccc(Cc2cc[nH+]cc2-c2cccc(F)c2)ccc1C[C@H]1CCN1O', 'Cn1cc[n+](-c2cccc(C(F)(F)F)c2)c2cccc1-2', 'O=C1c2c(O)cc(O)cc2C[C@@H]1Oc1ccc(F)cc1CO', 'O=c1ccn(Cc2ccncc2CF)cc1-c1ccc(O)cc1O', '[NH3+]Cc1cncnc1N1CCN(c2ccn(CF)c(=O)c2)CC1', 'O=c1cnc(Cn2cccc2Nc2ccnc(CF)c2)ccc1NO', 'O=C1C(=O)c2ccc(/C=C\\Cc3ccsc3)cc2C=C1[O-]', 'O=c1ccn(-n2cc(C(F)F)nc2CO)cc1-c1ccc(CO)c(F)c1', 'O=c1ccn(NCc2ccccc2F)cc1-n1ccc(CF)c1', 'O=C1C=C(OCc2cc(O)cc(O)c2)C=C2C=C(c3cccnc3)C=CN1O2', 'O=c1ccc(Cn2cc(CO)cc2F)cc2cc(O)ccc12', 'CCc1cc(Cn2cccc2Cn2ccc(CF)c2)nccc1=O', 'OCc1c[nH+]c(N2CCN(c3ccccc3CF)C[C@@H]2F)s1', 'O=c1cc(F)nc2ccc(Cc3ccc(O)cc3)cc2c1F', 'O=c1cnc(Nn2c[nH+]cc2-c2ccccc2F)ccc1F', 'O=c1ccn(Cc2ccn(CO)c(=O)c2)c2c(O)cccc12', 'O=c1cnn(-n2c[nH+]cc2-c2cccnc2)cc1CSO', '[NH3+][C@@H]1C[C@@H](CC/C=C/c2ccccc2-c2cccc(O)c2)N=N1', 'Nc1ccc2c([nH+]1)c(=O)ccn1cc(C(F)(F)F)cc21', 'O=c1cnc(Cn2cnc3cccn32)ccc1[C@H]1CCC=C1CO', 'O=c1[nH]cc(-c2ccncc2)cc1-c1ccn(CO)c1CCl', 'ONc1ccnc(Cc2ccccc2-c2cccc(CF)c2)c1F', 'CO[S@](=O)c1ccccc1C1=CC=Cc2c(cn(C(F)F)c2[O-])C1', 'O=c1cnc(Nc2cc[nH+]cc2-c2cnco2)ccc1N1CC[N@H+]1[O-]', 'O=c1ccc(Cn2c[nH+]cc2[O-])cc(-c2ccccc2O)c1F', 'O=c1cnc(Cn2cncc2Cn2ccc(F)c2)ccc1CCl', 'O=c1cnc(-c2cccc(O)c2)cc2cc(O)ccc12', 'O=c1ccc(CF)nn1Nc1ccccc1-c1cccc(O)c1', 'O=c1ccn(Cn2c[nH+]cc2Cc2ccsn2)cc1N1CCCCO1', 'O=C1C=CN(Cc2ccc(CO)cn2)C2=CC(CO)=CC=C1S2', 'CCc1c[nH+]ccc1/N=C/c1ccc(OCn2cc[nH+]c2)cc1C1CC1', 'O=c1ccn(-c2ccnc(F)c2)nc1N1CC[N@@H+](CO)C1', 'OSc1cc[nH+]c(Cc2c[nH+]ccc2N2Cc3ccccc3O2)c1', 'O=c1cc(-c2ccc(F)cn2)cccc1CN1NOCOC[C@@H]1O', 'O=c1nc(CF)[nH]c2c(CF)c([C@@H](F)[C@@H](CO)Cc3ccccc3)sc12', 'OSCN1C(c2ccc(SO)cc2F)=CSc2cc(F)ccc21', 'O=CCc1ccc(-c2ccccc2CO)cc1Cc1ccccc1C=O', 'O=c1cc(F)cccc1Nc1cccc(-c2cncc(OCCF)c2)c1F', 'O=c1ccc(Cc2ccn(F)c2)ccc1-n1cnc(CF)n1', 'O=c1ccn(-n2cc(CF)cc2-n2cc[nH+]c2)cc1CCF', 'O=c1ccn(CCc2ccccc2CF)cc1N1CCN2OCCN21', 'O=c1cc(-c2cccc(O)c2)cccc1-c1cnn(O)c1', 'O=c1ccc(Cn2cncc2-n2ccc(CF)c2)ccc1CO', 'Cc1cc2ccc(SCC(=O)N3COCOCNN3)n2n1O', 'O=c1cnn2c(CF)c[nH+]c2c2ccc(CF)cc12', 'O=c1cc(-c2ccnn2CF)cccc1-c1ccc(F)cn1', 'O=c1cc(-c2ccn(CO)c2-c2ccccc2O)cccc1NO', 'O=C1C=CN(c2ccccc2N2CCN(c3ccccc3)C2)C1', 'OCc1cc(N2CCN(c3cccc(O)c3)CC2)ccc1[C@H](O)F', 'O=C(CNCc1cc(=O)cccc1F)N1N=CNCc2ccc[nH+]c21', 'O=c1cc(-n2cccc2Oc2ccn(CF)c2F)cccc1CO', 'OCc1ccc[nH+]c1-c1ccoc1CCON1N=CCCc2cccnc21', 'O=c1ccn(Oc2cccc(CO)c2)cc1-n1cccc1Cl', 'Oc1cc[n+](Cc2cccnc2Cc2ccoc2F)cc1N1CCN1', '[NH3+][C@@H](O)Cc1nn(Cc2cccc(-c3cccc(F)c3)c2)ncc1=O', 'O=c1ccn(-n2ccc(F)n2)cc1-c1cc(-c2ccccc2)ccc1CO', 'O=c1ccn(Cn2cc[nH+]c2-c2ccsc2)cc1-c1ccon1', 'N[C@@H]1[NH2+]CCN1C(=O)c1cn(-c2ccccc2)c(C(F)F)n1', 'OCc1cc(F)ccc1-n1cc[nH+]c1-c1ccc(F)cc1CCl', 'O=c1ccn(Cc2ccn(CO)c2)cc1-n1ccc(F)c1', 'OCCCc1cc(CO)ccc1N1CC=N[C@H](c2cccs2)C=N1', 'O=c1cnc(-n2ccc(F)c2Cl)cnc1N1CCC[C@H]1O', 'O=c1ccc(-c2ccsc2)ccc1-n1ccc(CO)c1CO', 'O=c1cc(-c2cc[nH+]cc2)ccc(SCc2cccnc2)c1F', 'NC(=O)[C@H](O)Nc1ccc2cc(C(F)(F)F)ccc2c1', 'O=c1scccc1N1CCN(c2cncn2CCO)[C@H]1CF', 'O=c1cnc(Cc2cnncc2-c2ccn(CF)c2)ccc1F', 'O=c1ccn(CNc2ccncc2F)cc1-n1ccc(Br)c1', 'CCN1CCC=C(C2=CCc3c2nnc(-c2ccc(O)cc2CO)cc3=O)O1', 'O=c1[nH]cc(Cn2ccc(CF)c2F)cc1-c1ccccc1', 'O=c1cnc(-c2ccnn2-c2cccn2CO)cc(F)c1F', 'O=c1ccc(-c2cccc(CO)c2)ccc1-c1ccncc1', 'N[C@@H](O)CN1Cc2c(F)cccc2-c2cc[nH]c2C1=O', 'O=C1OCCN1c1cn(Cc2ccnnc2-c2ccsc2)ccc1=O', 'Nc1ncn(-c2ccc(Nc3cc[nH]c(=O)c3)ccc2=O)n1', 'O=c1cc(-c2cccc(O)c2)cccc1-n1cccc1S', 'OSc1ccccc1CC1=C(Oc2ccccc2-n2cccc2F)O1', 'O=c1cncc(Cn2cccc2NNc2ncccc2F)cc1CF', 'O=c1ccn(Cn2ccnc2CO)cc1-n1cnc(CO)c1F', 'O=C1NCCN1c1cn(Cc2ccoc2-n2ccs2)ccc1=O', 'O=Cc1ccccc1N1N=C[C@@H](N2Cc3ccc[nH+]c32)SC1=O', 'O=c1ccc(Cn2c[nH+]cc2-c2cnnc(CF)c2)ccc1O', 'O=c1ccc(-c2ccn(CO)c2CO)ccc1-c1ccsc1', 'O=c1cncc(Cn2cccc2-c2cnsc2CF)cc1CCF', 'O=C([O-])c1cccc2sc(-c3ccsc3-c3cccc(F)c3)cc(=O)c12', 'NCn1cccc1-c1c[n+](Cc2ccccc2Cl)coc1=O', 'ONc1ccc(-c2ccncc2F)cc1-c1ccc(O)cc1Cl', 'O=C1/C=C(c2cc[nH+]cc2)\\C=C(\\c2ccccc2)O/[N+]([O-])=N\\1', 'OCCc1c(F)cccc1C/[NH+]=c1\\cc2ccccc2ccc1NO', 'O=c1nc(-c2cn(-c3cccc(CO)c3)ccc2=O)cn[nH]1', 'O=c1ccn(OCc2ccn(CO)c2)cc1-c1ccccc1F', 'O=c1cc(-c2ccn(CO)c2)cc2ccc(C(F)F)cn12', 'O=[SH]C1=CC=CC=CN1CC1=C(O)[C@@H](F)[N+](c2cccnc2)=C1', 'NCc1nccn1Cn1ccc(=O)c2c1CC=CC1=CN[C@@H]12', '[NH3+][C@H](CO)c1ccc(/C=C2\\Nc3cc(F)cnc3C2=O)cc1CC(F)F', 'O=c1ccn(CCn2ccc(CO)c2)nc1-c1ccccc1F', 'NC(=O)[C@@H](O)/N=C1\\Cc2ccccc2[C@@H]1c1cccs1', 'O=c1ccn(Nn2ccc(F)c2-n2cc(F)cc2CF)nc1-c1ccccc1', 'O=c1ccc(-c2ccn(CF)c2)ccc1OCCn1ccsc1=O', 'O=c1ccn(Nn2nccc2-c2ccoc2)cc1-n1cccn1', 'O=c1ccn(-c2ccccc2F)c(CSc2ccncc2CF)n1', 'OCc1ccnn1-c1cc(Cl)cc(-c2ccc(F)cn2)c1', 'CCc1cncn1S(=O)(=O)C[N@@H+]1CCN(c2ccc(C)cc2)C1', 'O=c1cc(BO)cc2ccc(-c3ccccc3CO)cc2c1F', 'OCC1=CC2=CC=C(O1)/C(=C/c1ccc(S(O)(F)F)cc1)N2', '[NH3+]Cc1cn(-c2ccc(O)cc2CO)cc1Cc1ccccc1Cl', 'CN1NC(=O)N(C(=O)c2cccnc2)N=C1c1cccc(C(N)=O)c1Cl', '[NH3+][S@@](O)(CCF)Nc1[nH+]c(Cc2cccc(O)c2)ccc1CO']
    for smile in smiles_test:
        print(compute_properties(smile))
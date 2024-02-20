import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
import networkx as nx
import copy
import os
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
import SA_Score.sascorer as SA

def read_frags_from_csv():
    frag_path = './fragments.csv'

    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    mol_list = []
    name_list = []
    smi_list = df_frag['SMILES'].to_list()
    name_list = df_frag['frag name'].to_list()
    for num in range(len(smi_list)):
        smi = smi_list[num]
        name = name_list[num]
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            mol_list.append(mol)
            name_list.append(name)
        else:
            print('@read_frags_from_csv: mol is none for %s' %(name))
    return mol_list

def generate_start_pop_hexane(pop_size):
    smi = 'CCCCCC'
    start_pop_list = []
    mol = Chem.MolFromSmiles(smi)
    smi_list = []

    for i in range(pop_size):
        start_pop_list.append(mol)
        smi_list.append(smi)

    data = {'mol': start_pop_list, 'smi': smi_list}
    df_pop = pd.DataFrame(data=data)
    df_pop['fitness value'] = ''
    df_pop['SA score'] = ''

    df_pop['prop'] = df_pop['mol'].apply(calc_prop)

    df_pop['SA score'] = df_pop['mol'].apply(SA.calculateScore)

    # get fitness vals
    df_pop = get_fitness_vals(df_pop)

    return df_pop

def generate_random_start_pop(pop_size, SA_max, max_mol_weight):

    if not os.path.exists('random_start_pop.csv'):
        df_pop = pd.read_csv('mcule_preselection.csv', sep=',')
        df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)
        df_pop['SA score'] = df_pop['mol'].apply(SA.calculateScore)
        df_pop['Mol wt'] = df_pop['mol'].apply(Descriptors.MolWt)
        drop_idxs = df_pop.index[df_pop['SA score'] > SA_max]
        df_pop = df_pop.drop(drop_idxs)
        drop_idxs = df_pop.index[df_pop['Mol wt'] > max_mol_weight]
        df_pop = df_pop.drop(drop_idxs)
        df_pop = df_pop.sample(pop_size)
        df_pop['fitness value'] = ''
        df_pop['prop'] = df_pop['mol'].apply(calc_prop)

        # get fitness vals
        df_pop = get_fitness_vals(df_pop)
        df_pop.to_csv('random_start_pop.csv')

    else:
        df_pop = pd.read_csv('random_start_pop.csv')
        df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)

    return df_pop


def get_valences(mol):
    # implicit valences
    num_atoms = mol.GetNumAtoms()
    atom_idx_list = []
    atom_symbol_list = []
    valence_list = []

    for i in range(num_atoms):
        valence = mol.GetAtomWithIdx(i).GetImplicitValence()
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atom_idx_list.append(i)
        atom_symbol_list.append(symbol)
        valence_list.append(valence)
    return atom_idx_list, atom_symbol_list, valence_list

def get_total_valences(mol):
    # implicit valences
    num_atoms = mol.GetNumAtoms()
    atom_idx_list = []
    atom_symbol_list = []
    valence_list = []

    for i in range(num_atoms):
        valence = mol.GetAtomWithIdx(i).GetTotalValence()
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atom_idx_list.append(i)
        atom_symbol_list.append(symbol)
        valence_list.append(valence)
    return atom_idx_list, atom_symbol_list, valence_list

def get_explicit_valences(mol):
    # implicit valences
    num_atoms = mol.GetNumAtoms()
    atom_idx_list = []
    atom_symbol_list = []
    valence_list = []

    for i in range(num_atoms):
        valence = mol.GetAtomWithIdx(i).GetExplicitValence()
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atom_idx_list.append(i)
        atom_symbol_list.append(symbol)
        valence_list.append(valence)
    return atom_idx_list, atom_symbol_list, valence_list

def deletion(mol):
    """
    performs atom deletion on rdkit mol object

    checks if number of atoms is > 1 to ensure that no empty molecule results

    the networkx package returns the atom indices which - if deleted - result in a splitted molecule
    these atoms are not open for deletion. any other atom can be deleted (randomly chosen)

    returns rdkit mol object

    author: laura koenig-mattern

    """

    flag_successful = 0

    start_mol = copy.deepcopy(mol)

    if mol.GetNumAtoms() > 1:  #note for older implementation: it is possible to set mol weight as limit cause otherwise graphs get too large and nx.all_cut_sets needs hours

        atom_idx_set = set()
        for atom in mol.GetAtoms():
            atom_idx_set.add(atom.GetIdx())


        adj_matrix = Chem.GetAdjacencyMatrix(mol)

        # get sets of nodes (-> atoms) that, if deleted, split the molecule into two or more fragments
        # articulation points
        graph = nx.Graph(adj_matrix)
        mol_idx_set = set(list(nx.articulation_points(graph)))
        removable_atom_idx = list(atom_idx_set.difference(mol_idx_set))

        if len(removable_atom_idx) == 0:
            flag_successful = 0
            del_mol = start_mol
            del_mol.UpdatePropertyCache()
            Chem.SanitizeMol(del_mol)
            return del_mol, flag_successful

        rand_int = np.random.randint(len(removable_atom_idx))


        rm_idx = removable_atom_idx[rand_int]

        Chem.Kekulize(mol, clearAromaticFlags=True)

        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAtom(rm_idx)

        del_mol = rw_mol.GetMol()
    else:
        del_mol = start_mol
        del_mol.UpdatePropertyCache()
        Chem.SanitizeMol(del_mol)


    if start_mol.GetNumAtoms() > del_mol.GetNumAtoms():
        flag_successful = 1

    if del_mol is not None:
        del_mol.UpdatePropertyCache()
        Chem.SanitizeMol(del_mol)


    return del_mol, flag_successful

def fragmentor(mol):
    """
    splits a given molecule into fragments based on bridge connections

    :param mol: rdkit mol object to be fragmented
    :return: frag_mol: fragmented molecule, flag_successful: 0 - operation not successful, 1 - operation successful

    author: laura koenig-mattern
    """

    flag_successful = 0
    num_bonds = mol.GetNumBonds()

    if num_bonds >= 1:
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        graph = nx.Graph(adj_matrix)

        # returns bridges of the graph as tuple of node indices stored in list
        bridges = [i for i in nx.bridges(graph)]

        if len(bridges) >= 1:
            # get bond object of the bridges
            rand_int = np.random.randint(len(bridges))
            tup = bridges[rand_int]
            atom_idx_1 = tup[0]
            atom_idx_2 = tup[1]

            Chem.Kekulize(mol, clearAromaticFlags=True)

            rw_mol = Chem.RWMol(mol)
            rw_mol.RemoveBond(atom_idx_1, atom_idx_2)

            frag_mol = rw_mol.GetMol()

            if len(Chem.GetMolFrags(frag_mol)) > 1:
                flag_successful = 1

        else:
            # split ring to get fragmented mol
            frag_mol, flag_successful = split_ring(mol)

    else:
        frag_mol = mol
        flag_successful = 0

    frag_mol.UpdatePropertyCache()
    Chem.SanitizeMol(frag_mol)

    return frag_mol, flag_successful

def combinator(frag_mol):
    """
    randomly combines n given molecular fragments until all frags are connected

    :param frag_mol: rdkit mol object containing n fragments
    :return: comb_mol: rdkit mol object containing the connected mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """

    flag_successful = 0
    start_mol = copy.deepcopy(frag_mol)
    idx_tup = Chem.GetMolFrags(frag_mol)
    num_frags = len(idx_tup)

    counter = 0

    while num_frags > 1:
        Chem.Kekulize(frag_mol, clearAromaticFlags=True)
        frag_mol.UpdatePropertyCache()

        atom_idx_list, atom_symbol_list, valence_list = get_valences(frag_mol)

        val_tup = []
        for tup in idx_tup:
            v_tup = tuple([valence_list[element] for element in tup])
            val_tup.append(v_tup)

        # eligable atom idices (eligable means valence is larger than 1 -> eligable for bonding)
        eligable_idx = []

        count_non_eligable_frags = 0

        for num, tup in enumerate(val_tup):
            arr = np.asarray(tup)
            nonzero_idx = np.flatnonzero(arr)
            atm_idxs_elig = np.array(idx_tup[num])[list(nonzero_idx)]
            eligable_idx.append(atm_idxs_elig.tolist())
            if len(list(nonzero_idx)) == 0:
                count_non_eligable_frags = count_non_eligable_frags + 1

        if count_non_eligable_frags > 0:
            #print('combination fails. Number of eligable fragments < number of fragments')
            comb_mol = frag_mol
            return comb_mol, flag_successful

        rng = np.random.default_rng()
        frag_idx = rng.choice(num_frags, 2, replace=False)

        rand_int_1 = np.random.randint(len(eligable_idx[frag_idx[0]]))
        atm_idx1 = eligable_idx[frag_idx[0]][rand_int_1]
        val_atm1 = valence_list[atm_idx1]
        rand_int_2 = np.random.randint(len(eligable_idx[frag_idx[1]]))
        atm_idx2 = eligable_idx[frag_idx[1]][rand_int_2]
        val_atm2 = valence_list[atm_idx2]


        rw_mol = Chem.RWMol(frag_mol)


        if val_atm1 <= 1 or val_atm2 <= 1:
            rw_mol.AddBond(atm_idx1, atm_idx2, order=Chem.BondType.SINGLE)
            #print('bond added')
        if val_atm1 > 1 and val_atm2 > 1:
            rand_int = np.random.randint(2)
            if rand_int == 0:
                rw_mol.AddBond(atm_idx1, atm_idx2, order=Chem.BondType.SINGLE)
                #print('bond added')
            if rand_int == 1:
                rw_mol.AddBond(atm_idx1, atm_idx2, order=Chem.BondType.DOUBLE)
                #print('bond added')

        frag_mol = rw_mol.GetMol()
        frag_mol.UpdatePropertyCache()
        idx_tup = Chem.GetMolFrags(frag_mol)
        num_frags = len(idx_tup)

        counter = counter + 1

        if counter > 100:
            break

    comb_mol = frag_mol

    if num_frags == 1:
        flag_successful = 1

    if num_frags > 1:
        flag_successful = 0
        comb_mol = start_mol

    if comb_mol is not None:
        comb_mol.UpdatePropertyCache()
        Chem.SanitizeMol(comb_mol)
    return comb_mol, flag_successful

def addition(mol1, mol2):
    '''
    connects two molecular fragments to obtain one mol
    :param mol1: rdkit mol obj of fragment 1
    :param mol2: rdkit mol obj of fragment 2
    :return: add_mol: rdkit mol obj of connected mol, flag successful: 0 - not successful, 1 - successful
    '''

    flag_successful = 0
    new_mol = Chem.CombineMols(mol1, mol2)
    add_mol, flag_successful_comb = combinator(new_mol)

    if len(Chem.GetMolFrags(add_mol)) == 1:
        flag_successful = 1

    add_mol.UpdatePropertyCache()
    Chem.SanitizeMol(add_mol)

    return add_mol, flag_successful

def relocation(mol):
    """
    fragments a given molecule into two or more fragments. randomly recombines these fragments
    to obtain a new molecules
    :param mol: rdkit mol obj containing the molecule for relocation
    :return: rel_mol: rdkit mol obj containing the relocated molecule, flagsuccessful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """

    flag_successful = 0
    frag_mol, flag_successful_frag = fragmentor(mol)
    rel_mol, flag_successful_comb = combinator(frag_mol)

    if flag_successful_comb == 1 and flag_successful_frag == 1:
        flag_successful = 1

    if flag_successful == 1:
        rel_mol.UpdatePropertyCache()
        Chem.SanitizeMol(rel_mol)


    return rel_mol, flag_successful

def insertion(mol1, mol2):
    """
    inserts one molecule into another molecule. the molecule to be inserted remains the same.
    the molecule to which it should be inserted, is fragmented. the fragments + the mol to be inserted are combined.
    the molecule to be fragmented/inserted is chosen randomly. before the operation is executed, the possible combinations are determined

    :param mol1: rdkit mol obj of one molecule
    :param mol2: rdkit mol obj of the other  molecule
    :return: ins_mol: rdkit obj of inserted mol, flag_successful: 0 - operation not successful, 1 - operation successful

    author: laura könig-mattern
    """

    frag_mol1, flag_successful1 = fragmentor(mol1)
    frag_mol2, flag_successful2 = fragmentor(mol2)


    atom_idx_list1, atom_symbol_list1, valence_list1 = get_valences(mol1)
    atom_idx_list2, atom_symbol_list2, valence_list2 = get_valences(mol2)

    inserted_frag_bool_1 = False
    inserted_frag_bool_2 = False
    insertion_possible_bool_1 = False
    insertion_possible_bool_2 = False

    if np.any(valence_list1):
        inserted_frag_bool_1 = True
    if np.any(valence_list2):
        inserted_frag_bool_2 = True
    if flag_successful1 == 1:
        insertion_possible_bool_1 = True
    if flag_successful2 == 1:
        insertion_possible_bool_2 = True


    flag_successful_ins = 0
    flag_successful_frag = 0
    flag_successful = 0

    ins_mol = None

    if inserted_frag_bool_1 and not inserted_frag_bool_2 and not insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert frag one into frag two
        frag_mol = Chem.CombineMols(mol1, frag_mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1

    # 0 1 1 0
    if not inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and not insertion_possible_bool_2:
        # insert frag two into frag one
        frag_mol = Chem.CombineMols(frag_mol1, mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 0 1 1
    if inserted_frag_bool_1 and not inserted_frag_bool_2 and insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert frag 1 in frag 2
        frag_mol = Chem.CombineMols(mol1, frag_mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 0 1 1 1
    if not inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert frag 2 in frag 1
        frag_mol = Chem.CombineMols(frag_mol1, mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 1 1 0
    if inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and not insertion_possible_bool_2:
        # insert mol 2 into mol 1
        frag_mol = Chem.CombineMols(frag_mol1, mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 1 0 1
    if inserted_frag_bool_1 and inserted_frag_bool_2 and not  insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert mol 1 into mol 2
        frag_mol = Chem.CombineMols(mol1, frag_mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 1 1 1
    if inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and insertion_possible_bool_2:
        # choose both randomly
        rand_ins_int = np.random.randint(2)
        if rand_ins_int == 0:
            frag_mol = Chem.CombineMols(frag_mol1, mol2)
            ins_mol, flag_successful_ins = combinator(frag_mol)
            flag_successful_frag = 1
        if rand_ins_int == 1:
            frag_mol = Chem.CombineMols(mol1, frag_mol2)
            ins_mol, flag_successful_ins = combinator(frag_mol)
            flag_successful_frag = 1

    if flag_successful_frag == 1 and flag_successful_ins == 1 and ins_mol is not None:
        flag_successful = 1

    if ins_mol is not None:
        ins_mol.UpdatePropertyCache()
        Chem.SanitizeMol(ins_mol)

    return ins_mol, flag_successful

def bond_mutation(mol):
    """
    mutates bond type: single to double or double to single bond. bond is randomly chosen.
    if single bond: check valences if mutation to double bond is possible

    :param mol: rdkit mol obj containing the mol to be mutated
    :return: mol: rdkit mol obj containing mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    flag_successful = 0

    ini_mol = copy.deepcopy(mol)

    bonds = [bond for bond in mol.GetBonds()]
    atom_idx_list, atom_symbol_list, valence_list = get_valences(mol)
    num_bonds = mol.GetNumBonds()
    rand_bond_idxs = np.random.choice(num_bonds, num_bonds, replace=False)

    for num in range(num_bonds):
        rand_bond_idx = rand_bond_idxs[num]
        bond = bonds[rand_bond_idx]
        type = bond.GetBondType()


        atm_idx1 = bond.GetBeginAtomIdx()
        atm_idx2 = bond.GetEndAtomIdx()
        val1 = valence_list[atm_idx1]
        val2 = valence_list[atm_idx2]

        if str(type) == 'SINGLE':
            if val1 >= 1 and val2 >= 1:
                bond.SetBondType(Chem.BondType.DOUBLE)
                flag_successful = 1
                break

        if str(type) == 'DOUBLE':
            bond.SetBondType(Chem.BondType.SINGLE)
            flag_successful = 1
            break

        if str(type) == 'AROMATIC':
            copy_mol = copy.deepcopy(mol)
            Chem.Kekulize(copy_mol, clearAromaticFlags=True)
            copy_bonds = [bond for bond in copy_mol.GetBonds()]
            copy_bond = copy_bonds[rand_bond_idx]
            copy_type = copy_bond.GetBondType()

            if str(copy_type) == 'DOUBLE':
                Chem.Kekulize(mol, clearAromaticFlags=True)
                bond.SetBondType(Chem.BondType.SINGLE)
                flag_successful = 1
                break

            if str(copy_type) == 'SINGLE':
                Chem.SanitizeMol(mol)
                mol = ini_mol

    if mol is not None:
        try:
            mol.UpdatePropertyCache()
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except:
            #import ipdb; ipdb.set_trace()
            mol.UpdatePropertyCache()
            Chem.Kekulize(mol, clearAromaticFlags=True)
        try:
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
        except:
            import ipdb; ipdb.set_trace()

    return mol, flag_successful

def selection(df_pop, num_parents):
    """
    selects num_parents fittest molecules based on roulette wheel selection for cross-over

    :param df_pop: dataframe of the population containing rdkit mol obj, MP, BP, property predictions, fitnes vals
    :param num_parents: number of mols to be chosen for cross over using roulette wheel selection
    :return: selected_idxs: loc-indices of the selected mols from df_pop

    author: laura koenig-mattern
    """

    fitness_vals = df_pop['fitness value'].to_numpy().reshape(df_pop['fitness value'].shape[0]).astype(np.float64)

    # shift fitness vals such that all are larger or equal than 1
    fitness_vals_new = fitness_vals + np.abs(np.min(fitness_vals)) + 1  # circumvent zero

    # calc probabilities
    probs = fitness_vals_new / np.sum(fitness_vals_new)
    selected_iloc_idxs = np.random.choice(len(fitness_vals), num_parents, p=probs)

    selected_idxs = df_pop.index[selected_iloc_idxs]

    return selected_idxs

def cross_over(df_pop, num_parents, num_children, max_mol_weight, group_constraints, SA_max):
    """
    performs cross-over. num_parents selected parent mols are fragmented, these fragments are the "mating pool"
    randomly choose fragments from the mating pool, to produce num_children child molecules
    :param df_pop: pandas dataframe of population
    :param num_parents: number of parent mols
    :param num_children: number of child mols
    :param max_mol_weight: maximum mol weight
    :param group_constraints: constraints on predefined functional groups
    :param SA_max: max synthetic accessibility
    :return: df_pop: pandas dataframe with pop + child mols
    """
    # select individuals based on roulette wheel selection
    selected_idxs = selection(df_pop, num_parents)

    # fragment the chosen molecules and add fragments to pool
    frag_pool = []
    for idx in selected_idxs:

        mol = df_pop.loc[idx, 'mol']

        frag_mol, flag_successful = fragmentor(mol)

        if flag_successful == 1:
            frags = Chem.GetMolFrags(frag_mol, asMols=True)

            for frag in frags:
                frag_pool.append(frag)

        # if flag_successful == 0:
        #    import ipdb; ipdb.set_trace()

    # form desired number of children
    num_children_formed = 0
    smi_children = []
    counter = 0
    while num_children_formed < num_children:
        # randomly choose two fragments from pool
        frag_idxs = np.random.choice(len(frag_pool), 2, replace=False)
        idx1 = frag_idxs[0]
        idx2= frag_idxs[1]
        frag1 = frag_pool[idx1]
        frag2 = frag_pool[idx2]
        new_mol = Chem.CombineMols(frag1, frag2)

        # combine
        comb_mol, flag_comb_successful = combinator(new_mol)
        if flag_comb_successful == 1:
            try:
                comb_mol.UpdatePropertyCache()
                Chem.SanitizeMol(comb_mol)

            except:
                import ipdb; ipdb.set_trace()
            if Chem.Descriptors.MolWt(comb_mol) > max_mol_weight:
                flag_comb_successful = 0
            if group_constraints == 'acid stability' or group_constraints == 'acid stable CO only' \
                    or group_constraints == 'acid stable CO only sugar':
                stability_bool = acid_stability_checker(comb_mol)
                if not stability_bool:
                    flag_comb_successful = 0
            if SA_max:
                SA_score = SA.calculateScore(comb_mol)
                if SA_score > SA_max:
                    flag_comb_successful = 0

        counter = counter + 1

        if flag_comb_successful == 1:
            num_children_formed = num_children_formed + 1
            smi_children.append(Chem.MolToSmiles(comb_mol))

        if counter > 100:
            break

    # get the smiles from children which are differnt from each other (make sure to add no duplets)
    smi_children_diff = list(set(smi_children))

    # only add to frame if not already in population
    residual_smi = smi_children_diff
    for i in range(len(smi_children_diff)):
        smi = smi_children_diff[i]
        #import ipdb; ipdb.set_trace()
        if not smi in df_pop['smi'].values:
            df_pop = df_pop.append(pd.Series(), ignore_index=True)
            loc_ind = df_pop.index[df_pop.shape[0]-1]
            df_pop.loc[loc_ind,'smi'] = smi
            mol = Chem.MolFromSmiles(smi)  #automatically removes all Hs
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            df_pop.loc[loc_ind, 'mol'] = mol

    return df_pop
def get_SA_scores(df_pop):
    """
    get synthetic accessibility score (SAS, see Ertl et al.)

    :param df_pop: pandas dataframe of the population
    :return: df_pop: pandas dataframe of the population updates with SAS

    author: laura könig-mattern
    """
    df_pop['SA score'] = df_pop['mol'].apply(SA.calculateScore)
    return df_pop

def calc_prop(mol):
    """
    property prediction
    as an example, we use rdkit's logP calculation
    :param mol: rdkit mol obj
    :return: prop: value of property of interest (logP)

    author: laura koenig-mattern
    """
    prop = Descriptors.MolLogP(mol)
    #prop = Chem.Crippen.MolLogP(mol)
    return prop

def calc_ring_penalty(mol):
    """
    calculates ring penalty of a molecule. the ring penalty s defined as numbers of rings with more than 6 atoms
    :param mol: rdkit mol ob
    :return: ring penalty
    """
    ri = mol.GetRingInfo()

    penalised_rings = []
    for ring in ri.AtomRings():
        if len(ring) > 6:
            penalised_rings.append(1)

    if len(penalised_rings) > 0:
        ring_penalty = sum(penalised_rings)
    else:
        ring_penalty = 0

    return ring_penalty

def get_fitness_vals(df_pop):
    """
    define your fitness_function here!
    as an example, we use rdkit's logP function, SAS and a ring penalty

    :param df_pop: pandas dataframe of population
    :return: df_pop: pandas dataframe of population, updated values
    """

    df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)
    df_pop['prop'] = df_pop['mol'].apply(calc_prop)
    df_pop['fitness value'] = df_pop['prop']

    # fitness function frequently used for benchmarking: lopP - SAS - Ringpenalty
    df_pop['fitness value'] = df_pop['prop'] - df_pop['mol'].apply(SA.calculateScore) - \
                              df_pop['mol'].apply(calc_ring_penalty)

    return df_pop
def get_random_frag():
    """
    draw a random fragment from a pool of fragments
    :return: frag: randomly drawn fragment

    author: laura koenig-mattern
    """
    frag_path = './fragments.csv'

    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    rand_iloc = np.random.randint(0,df_frag.shape[0])
    rand_loc = df_frag.index[rand_iloc]

    smi = df_frag.loc[rand_loc, 'SMILES']
    frag = Chem.MolFromSmiles(smi)

    return frag

def get_random_frag_ins():
    """
    draw a random atom for atom insertion
    :return: frag: randomly drawn atom
    """

    frag_path = './fragments_insert_atom.csv'

    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    rand_iloc = np.random.randint(0,df_frag.shape[0])
    rand_loc = df_frag.index[rand_iloc]

    smi = df_frag.loc[rand_loc, 'SMILES']
    frag = Chem.MolFromSmiles(smi)

    return frag

def get_random_frag_from_class():
    """
    randomly draws a class, and then, randomly draws a fragment from the class

    :return: frag: randomly drawn fragment

    author: laura koenig-mattern
    """

    frag_path = './fragments_group_wise.csv'
    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    rand_col = np.random.randint(0, df_frag.shape[1])

    df_col = df_frag.iloc[:,rand_col].dropna()

    rand_iloc = np.random.randint(0,df_col.shape[0])
    rand_loc = df_col.index[rand_iloc]

    smi = df_col.loc[rand_loc]
    frag = Chem.MolFromSmiles(smi)

    return frag

def bond_addition(mol):
    """
    adds a new bond between randomly chosen atoms

    :param mol: rdkit mol obk of mol to be mutated
    :return: add_bond_mol: rdkit mol obj of mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    flag_successful = 0
    atom_idx_list, atom_symbol_list, valence_list = get_valences(mol)

    # molecule must have at least two atoms to be able to create a bond
    if len(atom_idx_list) <= 1:
        flag_successful = 0
        add_mol_bond = mol
        return add_mol_bond, flag_successful

    # check valences to see which atoms are able to create bonds.
    eligable_atom_idx_list = []
    eligable_valence_list = []
    is_aromatic_list = []
    for num, valence in enumerate(valence_list):
        if mol.GetAtomWithIdx(atom_idx_list[num]).GetIsAromatic():
            is_aromatic_list.append(atom_idx_list[num])
        if valence >= 1:
            eligable_atom_idx_list.append(atom_idx_list[num])
            eligable_valence_list.append(valence)

    adj_matrix = Chem.GetAdjacencyMatrix(mol)

    # fill values in adj matrix where both atoms are aromatic with other value than zero or one
    # use is aromatic list
    for idx in is_aromatic_list:
        adj_matrix[idx][is_aromatic_list] = 2

    # fill diagonal values of adjacency matrix with other value than zero or one
    np.fill_diagonal(adj_matrix, 2)
    eligable_adj_matrix = adj_matrix[eligable_atom_idx_list,:][:,eligable_atom_idx_list]


    # find all zero arguments
    eligable_cons = np.argwhere(eligable_adj_matrix == 0)
    if eligable_cons.shape[0] == 0:
        add_mol_bond = mol
        flag_successful = 0
        return add_mol_bond, flag_successful

    # draw a random number to choose which bond to create
    rand_num = np.random.randint(eligable_cons.shape[0])
    con = eligable_cons[rand_num]
    atm_idx_0 = eligable_atom_idx_list[con[0]]
    val_0 = valence_list[atm_idx_0]
    atm_idx_1 = eligable_atom_idx_list[con[1]]
    val_1 = valence_list[atm_idx_1]

    # choose type of bond and add bond
    rw_mol = Chem.RWMol(mol)
    Chem.Kekulize(rw_mol, clearAromaticFlags=True)

    if val_0 == 1 or val_1 == 1:
        rw_mol.AddBond(atm_idx_0, atm_idx_1, order=Chem.BondType.SINGLE)
    if val_0 > 1 and val_1 > 1:
        rand_bond_int = np.random.randint(2)
        if rand_bond_int == 0:
            rw_mol.AddBond(atm_idx_0, atm_idx_1, order=Chem.BondType.SINGLE)
        if rand_bond_int == 1:
            rw_mol.AddBond(atm_idx_0, atm_idx_1, order=Chem.BondType.DOUBLE)

    add_bond_mol = rw_mol.GetMol()
    add_bond_mol.UpdatePropertyCache()
    Chem.SanitizeMol(add_bond_mol)

    flag_successful = 1

    return add_bond_mol, flag_successful

def bond_deletion(mol):
    """
    delete existing bond without splitting the molecule

    :param mol: rdkit mol object of mol for mutation
    :return: del_bond_mol: rdkit mol object of mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    flag_successful = 0

    num_bonds = mol.GetNumBonds()
    if num_bonds < 1:
        del_bond_mol = mol
        flag_successful = 0
        return del_bond_mol, flag_successful

    # get bonds
    bonds = mol.GetBonds()

    # get bridges
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    graph = nx.Graph(adj_matrix)
    bridge_atoms = [i for i in nx.bridges(graph)]

    bridges = []
    for i in bridge_atoms:
        atm_idx_0 = i[0]
        atm_idx_1 = i[1]
        bond_idx_curr = mol.GetBondBetweenAtoms(atm_idx_0, atm_idx_1).GetIdx()
        bridges.append(bond_idx_curr)


    erasable_bonds = []
    for bond_idx in range(num_bonds):
        if bond_idx not in bridges:
            erasable_bonds.append(bond_idx)

    if len(erasable_bonds) == 0:
        del_bond_mol = mol
        flag_successful = 0
        return del_bond_mol, flag_successful

    # randomly choose bond to delete
    rand_num = np.random.randint(len(erasable_bonds))
    erase_bond_idx = erasable_bonds[rand_num]
    atm_idx_0 = bonds[erase_bond_idx].GetBeginAtomIdx()
    atm_idx_1 = bonds[erase_bond_idx].GetEndAtomIdx()

    rw_mol = Chem.RWMol(mol)
    Chem.Kekulize(rw_mol, clearAromaticFlags=True)

    rw_mol.RemoveBond(atm_idx_0, atm_idx_1)

    del_bond_mol = rw_mol.GetMol()
    del_bond_mol.UpdatePropertyCache()
    Chem.SanitizeMol(del_bond_mol)

    flag_successful = 1

    return del_bond_mol, flag_successful

def substitution(mol):
    """
    substitute atom by other atom

    :param mol: rdkit mol obj of mol for mutation
    :return: subs_mol: rdkit mol obj of mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    copy_mol = copy.deepcopy(mol)
    flag_successful = 0

    symbol_subs = ['C', 'O', 'N', 'S', 'P']
    subs_valences = [4, 2, 3, 6, 5]
    atomic_nums = [6, 8, 7, 16, 15]

    # get atoms and explicit valences
    atoms = copy_mol.GetAtoms()
    atom_idx_list, atom_symbol_list, valence_list = get_explicit_valences(copy_mol)


    # try maximum 10 times
    count = 0
    for tries in range(10):
        count = count + 1
        # pick random atom from mol to be substituted
        rand_num = np.random.randint(len(atom_symbol_list))

        rand_idx = atom_idx_list[rand_num]
        atom = atoms[rand_idx]
        symbol = atom_symbol_list[rand_idx]
        valence = valence_list[rand_idx]

        # get possible subs
        possible_subs_idx = []
        for i in range(len(symbol_subs)):
            if symbol_subs[i] != symbol and subs_valences[i] >= valence:
                possible_subs_idx.append(i)

        if len(possible_subs_idx) == 0:
            continue

        # chose randomly from possible subs
        rand_int = np.random.randint(len(possible_subs_idx))
        subs_idx = possible_subs_idx[rand_int]
        atomic_num = atomic_nums[subs_idx]

        Chem.Kekulize(copy_mol, clearAromaticFlags=True)

        atom.SetAtomicNum(atomic_num)
        subs_mol = copy_mol

        try:
            subs_mol.UpdatePropertyCache()
            Chem.SanitizeMol(subs_mol)
            flag_successful = 1
        except:
            flag_successful = 0
            #import ipdb; ipdb.set_trace()
            break

        if flag_successful == 1:
            break

    if flag_successful == 0:
        subs_mol = mol

    return subs_mol, flag_successful

def get_neighbor_bonds(mol, atm_idx):
    """
    get bond idxs of neighboring bonds of an atom

    :param mol: rdkit mol obj
    :param atm_idx: atom idx of mol
    :return: bond_idxs: neighbor bond indices

    author: laura koenig-mattern
    """
    atom = mol.GetAtomWithIdx(atm_idx)
    bond_idxs = []
    for nbr in atom.GetNeighbors():
        bond_idxs.append(mol.GetBondBetweenAtoms(atm_idx, nbr.GetIdx()).GetIdx())
    return bond_idxs

def split_ring(mol):
    """
    delete a randomly chosen bond to split a ring.
    if several rings: randomly chose ring. predetermine deletable bonds.


    :param mol: rdkit mol obj of mol with ring to be split
    :return: frag_mol: rdkit mol obj with split ring

    author: laura koenig-mattern
    """
    flag_successful = 0

    ri = mol.GetRingInfo()
    bds_ring = []

    # get rings
    for ring in ri.BondRings():
        bds = set(ring)
        bds_ring.append(bds)

    # chose one ring
    rand_int = np.random.randint(len(bds_ring))
    this_ring = bds_ring[rand_int]

    intersect_bonds = []
    for i, ring in enumerate(bds_ring):
        if i != rand_int:
            intersect_bonds.append(this_ring.intersection(set(ring)))

    # bonds that are shared by multiple rings do not split the graph if deleted. therefore exclude
    eligable_bonds = list(this_ring)

    if len(intersect_bonds) > 0:
        for idx in this_ring:
            for bnd_set in intersect_bonds:
                if idx in bnd_set:
                    if idx in eligable_bonds:
                        eligable_bonds.remove(idx)

    if len(eligable_bonds) > 0:
        # randomly choose first bond
        bond_del_idx_0 = eligable_bonds[np.random.randint(len(eligable_bonds))]
        atm_idx_0 = mol.GetBondWithIdx(bond_del_idx_0).GetBeginAtomIdx()
        atm_idx_1 = mol.GetBondWithIdx(bond_del_idx_0).GetEndAtomIdx()

        # deleting neighbor bonds does not cut the ring into frags. Gets also non-ring-neighbors,
        # but that doesnt matter since we dont care about them when we want to split rings
        neighbor_bonds_0 = get_neighbor_bonds(mol, atm_idx_0)
        neighbor_bonds_1 = get_neighbor_bonds(mol, atm_idx_1)

        non_deletable = set(neighbor_bonds_0 + neighbor_bonds_1)

        eligable_bonds = list(set(eligable_bonds).difference(non_deletable))

    else:
        frag_mol = mol
        flag_successful = 0

    if len(eligable_bonds) > 0:
        bond_del_idx_1 = eligable_bonds[np.random.randint(len(eligable_bonds))]
        atm_idx_2 = mol.GetBondWithIdx(bond_del_idx_1).GetBeginAtomIdx()
        atm_idx_3 = mol.GetBondWithIdx(bond_del_idx_1).GetEndAtomIdx()

        copy_mol = copy.deepcopy(mol)
        Chem.Kekulize(copy_mol, clearAromaticFlags=True)

        rw_newmol = Chem.RWMol(copy_mol)
        rw_newmol.RemoveBond(atm_idx_0, atm_idx_1)
        rw_newmol.RemoveBond(atm_idx_2, atm_idx_3)

        frag_mol = rw_newmol.GetMol()

        if len(Chem.GetMolFrags(frag_mol)) > 1:
            flag_successful = 1
            frag_mol.UpdatePropertyCache()
            Chem.SanitizeMol(frag_mol)


        else:
            flag_successful = 0
            frag_mol = mol
    else:
        frag_mol = mol
        flag_successful = 0

    return frag_mol, flag_successful

def mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful, mol, mut_mol, SA_max):
    if flag_successful == 1:
        mol_wt = Chem.Descriptors.MolWt(mol)
        if mol_wt > max_mol_weight:
            flag_successful = 0
            mol = mut_mol
        if SA_max:
            SA_score = SA.calculateScore(mol)
            if SA_score > SA_max:
                flag_successful = 0
                mol = mut_mol

    return flag_successful, mol

def mol_cleaner(mol):
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    mol.UpdatePropertyCache()
    Chem.SanitizeMol(mol)
    return mol

def mutation(df_pop, mutation_rate, max_mol_weight, group_constraints, SA_max):
    """
    randomly choose mutation and perform the chosen operation.

    :param df_pop: pandas dataframe of the population
    :param mutation_rate: mutation rate (float)
    :param max_mol_weight: maximum mol weight (float)
    :param group_constraints: group constraints (str)
    :param SA_max: maximum synthetic accessibility score (float)
    :return: df_pop: pandas dataframe with updated mutated mols

    author: laura koenig-mattern
    """

    # iterate over each molecule in population
    for ilocInd in range(df_pop.shape[0]):
        locInd = df_pop.index[ilocInd]

        # mutation yes/no?
        rand_mut = np.random.random_sample()
        if rand_mut > mutation_rate:
            pass

        if rand_mut <= mutation_rate:

            rand_ints = np.array([0,1,2,3,4,5,6,7]) # each int for one operation
            np.random.shuffle(rand_ints)

            # get molecule that undergoes mutation
            mut_mol = df_pop.loc[locInd, 'mol']
            mut_mol_copy = copy.deepcopy(mut_mol)


            #import ipdb; ipdb.set_trace()

            # mutation. test each genetic operation until one was successful. the sequence of the operation is randomly chosen
            for rand_int in rand_ints:
                ## group addtion
                if rand_int == 0:
                    #print('addition')
                    #frag = get_random_frag()
                    frag = get_random_frag_from_class()
                    mol, flag_successful = addition(mut_mol_copy, frag)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful, mol, mut_mol, SA_max)

                ## relocation
                if rand_int == 1:
                    #print('relocation')
                    mol, flag_successful = relocation(mut_mol_copy)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful, mol, mut_mol, SA_max)


                ## atom addition
                if rand_int == 2:
                    #print('insertion')
                    #frag = get_random_frag()
                    frag = get_random_frag_ins()
                    mol, flag_successful = insertion(mut_mol_copy, frag)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful,
                                                                    mol, mut_mol,SA_max)

                ## atom deletion
                if rand_int == 3:
                    #print('deletion')
                    mol, flag_successful = deletion(mut_mol_copy)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful,
                                                                    mol, mut_mol, SA_max)


                ## bond type mutation
                if rand_int == 4:
                    #print('bond mutation')
                    mol, flag_successful = bond_mutation(mut_mol_copy)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful,
                                                                    mol, mut_mol, SA_max)


                ## bond addition
                if rand_int == 5:
                    #print('bond addition')
                    mol, flag_successful = bond_addition(mut_mol_copy)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful,
                                                                    mol, mut_mol, SA_max)

                ## bond deletion
                if rand_int == 6:
                    # print('bond addition')
                    mol, flag_successful = bond_deletion(mut_mol_copy)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful, mol, mut_mol, SA_max)
                    smi = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smi)

                ## atom substition
                if rand_int == 7:
                    # print('atom substitution')
                    mol, flag_successful = substitution(mut_mol_copy)
                    flag_successful, mol = mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful,
                                                                    mol, mut_mol, SA_max)

                if flag_successful == 1:
                    smi = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smi)
                    df_pop.loc[locInd, 'smi'] = smi
                    df_pop.loc[locInd, 'mol'] = mol

                    break


    return df_pop


def get_all_time_fittest(df_fittest, df_pop, num_fittest):
    df_fittest = df_fittest.append(df_pop, ignore_index=True)
    df_fittest = df_fittest.drop_duplicates(subset=['smi'])
    df_fittest = df_fittest.sort_values(by=['fitness value'], ascending=False)
    df_fittest = df_fittest.head(num_fittest)
    return df_fittest

def ini_statistics_frame():
    other_params = ['mean fitness', 'prop mean']
    col_names = other_params

    df_stats = pd.DataFrame(columns=col_names)

    return df_stats

def ini_group_frame():
    rd_groups = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
                 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
                 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
                 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
                 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
                 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    col_names = rd_groups

    df_groups = pd.DataFrame(columns=col_names)

    return df_groups

def get_stats(df_stats, df_pop):

    stat_dict = dict.fromkeys(['mean fitness', 'prop mean'])

    mean_fit = df_pop['fitness value'].mean()
    mean_prop = df_pop['prop'].mean()

    stat_dict['mean fitness'] = [mean_fit]
    stat_dict['prop mean'] = [mean_prop]

    df_app = pd.DataFrame.from_dict(stat_dict)

    df_stats = df_stats.append(df_app, ignore_index=True)

    return df_stats

def get_group_counts(df_groups, df_pop):
    rd_groups = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
                 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
                 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
                 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
                 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
                 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    group_dict = dict.fromkeys(rd_groups)
    for i in range(df_pop.shape[0]):
        locInd = df_pop.index[i]
        mol = df_pop.loc[locInd, 'mol']
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi)  # makes sure that Mol is sanitized

        for ii in rd_groups:
            func = getattr(Chem.Fragments, ii)
            num_groups_curr = func(mol)
            num_groups = group_dict[ii]

            if num_groups is None:
                num_groups = 0
            if isinstance(num_groups, list):
                num_groups = num_groups[0]


            tot_groups = num_groups + num_groups_curr


            group_dict[ii] = [tot_groups]

    df_app = pd.DataFrame.from_dict(group_dict)

    df_groups = df_groups.append(df_app, ignore_index=True)
    return df_groups

def post_process(df_pop, df_fittest, df_stats, df_groups, save_all_pops_flag, gen):

    ##### save frames #####
    save_path = './results/frames/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_all_pops_flag == 0:
        df_pop.to_csv(save_path + 'last_pop.csv')
    if save_all_pops_flag == 1:
        df_pop.to_csv(save_path + 'last_pop_' + str(gen) + '.csv')
    df_fittest.to_csv(save_path + 'fittest.csv')
    df_stats.to_csv(save_path + 'stats.csv')
    df_groups.to_csv(save_path + 'group_counts.csv')

    ##### print fittest molecules #####
    main_fig_path = './results/figures/'

    if not os.path.exists(main_fig_path):
        os.makedirs(main_fig_path)

    fig_path = main_fig_path + 'fittest.png'


    mol_list = df_fittest['mol'].to_list()
    props = df_fittest['prop'].to_list()

    #import ipdb; ipdb.set_trace()
    try:
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(200, 200),returnPNG=False, maxMols=500)
        img.save(fig_path)

    except:
        print('Drawing of Mol Grid not possible')

    ###### print mean fitness and mean of property #####
    gen_num = range(1,df_stats.shape[0]+1)
    mean_fitness = df_stats['mean fitness'].to_list()
    mean_prop = df_stats['prop mean'].to_list()

    fig_path = main_fig_path + 'mean_fit.png'
    plt.scatter(gen_num, mean_fitness, marker='o')
    plt.xlabel('number of generations')
    plt.ylabel('mean fitness')
    plt.savefig(fig_path)
    plt.close()

    fig_path = main_fig_path + 'mean_prop.png'
    plt.scatter(gen_num, mean_prop, marker='o')
    plt.xlabel('number of generations')
    plt.ylabel('mean prop')
    plt.savefig(fig_path)
    plt.close()

    ###### print group counts #####
    col_names = df_groups.columns

    for name in col_names:
        fig_path = main_fig_path + '%s.png' %(name)
        group_counts = df_groups[name].to_list()
        plt.scatter(gen_num, group_counts, marker='o')
        plt.xlabel('number of generations')
        plt.ylabel('group counts of %s' %(name))
        plt.savefig(fig_path)
        plt.close()
    return

def plot_mols(path_fittest):
    """
    draw mols on transparent background
    :param path_fittest: path to .csv with smiles of fittest mols
    :return: None

    author: laura koenig-mattern
    """
    fig_path = './results/figures/all_mols/'
    if not os.path.exists(fig_path):
        os.path.makedirs(fig_path)

    df_pop = pd.read_csv(path_fittest, sep=',')
    df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)
    mol_list = df_pop['mol'].to_list()

    for num in range(len(mol_list)):
        d2d = Draw.rdMolDraw2D.MolDraw2DCairo(200, 200)
        d2d.drawOptions().clearBackground = False
        # d2d.drawOptions().bgColour = (1,1,1,0)
        d2d.DrawMolecule(mol_list[num])
        d2d.FinishDrawing()
        d2d.WriteDrawingText(fig_path + '%s.png' % num)
    return

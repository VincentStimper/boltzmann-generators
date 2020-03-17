from collections import namedtuple
from itertools import chain
import numpy as np

basis_Zs = {}

# With Z-matrices - always make sure that atoms referenced in spots 2,3,4 have been referenced in spot 1 before.
# If the above does not happen it will throw an index error in the decompose Z indices function.
basis_Zs["ALA"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["HB1", "CB", "CA", "N"],
    ["HB2", "CB", "CA", "HB1"],
    ["HB3", "CB", "CA", "HB2"],
]  #

basis_Zs["LEU"] = [
    ["H", "N", "CA", "C"],  # improper ...
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD1", "CG", "CB", "CA"],  # torsion
    ["CD2", "CG", "CD1", "CB"],  # improper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG", "CG", "CD1", "CD2"],  # improper
    ["HD11", "CD1", "CG", "CB"],  # torsion methyl rotation
    ["HD21", "CD2", "CG", "CB"],  # torsion methyl rotation
    ["HD12", "CD1", "CG", "HD11"],  # improper
    ["HD13", "CD1", "CG", "HD12"],  # improper
    ["HD22", "CD2", "CG", "HD21"],  # improper
    ["HD23", "CD2", "CG", "HD22"],  # improper
]

basis_Zs["ILE"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG1", "CB", "CA", "N"],  # torsion
    ["CG2", "CB", "CA", "CG1"],  # improper
    ["CD1", "CG1", "CB", "CA"],  # torsion
    ["HB", "CB", "CA", "CG1"],  # improper
    ["HG12", "CG1", "CB", "CD1"],  # improper
    ["HG13", "CG1", "CB", "CD1"],  # improper
    ["HD11", "CD1", "CG1", "CB"],  # torsion methyl rotation
    ["HD12", "CD1", "CG1", "HD11"],  # improper
    ["HD13", "CD1", "CG1", "HD11"],  # improper
    ["HG21", "CG2", "CB", "CA"],  # methyl torsion
    ["HG22", "CG2", "CB", "HG21"],  # improper
    ["HG23", "CG2", "CB", "HG21"],  # improper
]

## NEED TO ADD CYM
basis_Zs["CYS"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["SG", "CB", "CA", "N"],  # torsion
    ["HB2", "CB", "CA", "SG"],  # improper
    ["HB3", "CB", "CA", "SG"],  # improper
    ["HG", "SG", "CB", "CA"],  # torsion
]

basis_Zs["CYM"] = [  # Disuflide
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["SG", "CB", "CA", "N"],  # torsion
    ["HB2", "CB", "CA", "SG"],  # improper
    ["HB3", "CB", "CA", "SG"],  # improper
]

basis_Zs["HIS"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD2", "CG", "CB", "CA"],  # torsion
    ["ND1", "CG", "CB", "CD2"],  # improper
    ["CE1", "ND1", "CG", "CB"],  # torsion but rigid
    ["NE2", "CD2", "CG", "ND1"],  # torsion but rigid
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HD2", "CD2", "CG", "NE2"],  # improper
    ["HE1", "CE1", "ND1", "CD2"],  # improper
    ["HE2", "NE2", "CD2", "CE1"],  # improper; epsilon protonated
    ["HD1", "ND1", "CG", "CE1"],  # improper; delta protonated
]

basis_Zs["ASP"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["OD1", "CG", "CB", "CA"],  # torsion
    ["OD2", "CG", "CB", "OD1"],  # torsion
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
]

basis_Zs["ASN"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["OD1", "CG", "CB", "CA"],  # torsion
    ["ND2", "CG", "CB", "OD1"],  # improper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HD21", "ND2", "CG", "CB"],  # torsion NH2
    ["HD22", "ND2", "CG", "HD21"],  # improper
]

basis_Zs["GLN"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD", "CG", "CB", "CA"],  # torsion
    ["OE1", "CD", "CG", "CB"],  # torsion
    ["NE2", "CD", "CG", "OE1"],  # torsion
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG2", "CG", "CB", "CD"],  # improper
    ["HG3", "CG", "CB", "CD"],  # improper
    ["HE21", "NE2", "CD", "CG"],  # NH2 torsion
    ["HE22", "NE2", "CD", "HE21"],  # improper
]

basis_Zs["GLU"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD", "CG", "CB", "CA"],  # torsion
    ["OE1", "CD", "CG", "CB"],  # torsion
    ["OE2", "CD", "CG", "OE1"],  # imppoper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG2", "CB", "CG", "CD"],  # improper
    ["HG3", "CB", "CG", "CD"],  # improper
]

basis_Zs["GLY"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA2", "CA", "N", "C"],
    ["HA3", "CA", "C", "N"],
]

basis_Zs["TRP"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD1", "CG", "CB", "CA"],  # torsion
    ["CD2", "CG", "CB", "CD1"],  # improper
    ["NE1", "CD1", "CG", "CB"],  # torsion but rigid
    ["CE2", "CD2", "CG", "CD1"],  # torsion but rigid
    ["CZ2", "CE2", "NE1", "CD1"],  # improper
    ["CH2", "CZ2", "CE2", "CD2"],  # torsion
    ["CZ3", "CH2", "CE2", "CD2"],  # improper
    ["CE3", "CD2", "CE2", "CD1"],  # improper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HD1", "CD1", "CG", "CD2"],  # improper
    ["HE1", "NE1", "CD1", "CD2"],  # improper
    ["HZ2", "CZ2", "CD2", "CZ3"],  # improper
    ["HH2", "CH2", "CZ3", "CD2"],  # improper
    ["HZ3", "CZ3", "CD2", "CZ2"],  # improper
    ["HE3", "CE3", "CD2", "CZ3"],  # improper
]

basis_Zs["TYR"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD1", "CG", "CB", "CA"],  # torsion
    ["CD2", "CG", "CB", "CD1"],  # improper
    ["CE1", "CD1", "CG", "CB"],  # torsion but rigid
    ["CE2", "CD2", "CG", "CD1"],  # torsion but rigid
    ["CZ", "CE1", "CE2", "CD1"],  # improper
    ["OH", "CZ", "CE1", "CE2"],  # improper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HD1", "CD1", "CG", "CE1"],  # improper
    ["HD2", "CD2", "CG", "CE2"],  # improper
    ["HE1", "CE1", "CD1", "CZ"],  # improper
    ["HE2", "CE2", "CD2", "CZ"],  # improper
    ["HH", "OH", "CZ", "CE1"],  # improper
]

basis_Zs["SER"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["OG", "CB", "CA", "N"],  # torsion
    ["HB2", "CB", "CA", "OG"],  # improper
    ["HB3", "CB", "CA", "OG"],  # improper
    ["HG", "OG", "CB", "CA"],  # torsion
]

basis_Zs["PRO"] = [
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD", "CG", "CB", "CA"],  # torsion
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG2", "CG", "CB", "CD"],  # improper
    ["HG3", "CG", "CB", "CD"],  # improper
    ["HD2", "CD", "CG", "N"],  # improper
    ["HD3", "CD", "CG", "N"],  # improper
]

basis_Zs["ARG"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD", "CG", "CB", "CA"],  # torsion
    ["NE", "CD", "CG", "CB"],  # torsion
    ["CZ", "NE", "CD", "CG"],  # torsion
    ["NH1", "CZ", "NE", "CD"],  # improper
    ["NH2", "CZ", "NE", "NH1"],  # improper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG2", "CG", "CB", "CD"],  # improper
    ["HG3", "CG", "CB", "CD"],  # improper
    ["HD2", "CD", "CG", "NE"],  # improper
    ["HD3", "CD", "CG", "NE"],  # improper
    ["HE", "NE", "CD", "CZ"],  # improper
    ["HH11", "NH1", "CZ", "NE"],  # improper
    ["HH12", "NH1", "CZ", "HH11"],  # improper
    ["HH21", "NH2", "CZ", "NE"],  # improper
    ["HH22", "NH2", "CZ", "HH21"],  # improper
]

basis_Zs["LYS"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD", "CG", "CB", "CA"],  # torsion
    ["CE", "CD", "CG", "CB"],  # torsion
    ["NZ", "CE", "CD", "CG"],  # torsion
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG2", "CG", "CB", "CD"],  # improper
    ["HG3", "CG", "CB", "CD"],  # improper
    ["HD2", "CD", "CG", "CE"],  # improper
    ["HD3", "CD", "CG", "CE"],  # improper
    ["HE2", "CE", "CD", "NZ"],  # improper
    ["HE3", "CE", "CD", "NZ"],  # improper
    ["HZ1", "NZ", "CE", "CD"],  # NH3 torsion
    ["HZ2", "NZ", "CE", "HZ1"],  # improper
    ["HZ3", "NZ", "CE", "HZ2"],  # improper
]

basis_Zs["MET"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["SD", "CG", "CB", "CA"],  # torsion
    ["CE", "SD", "CG", "CB"],  # torsion
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HG2", "CG", "CB", "SD"],  # improper
    ["HG3", "CG", "CB", "SD"],  # improper
    ["HE1", "CE", "SD", "CG"],  # torsion
    ["HE2", "CE", "SD", "HE1"],  # improper
    ["HE3", "CE", "SD", "HE2"],  # improper
]

basis_Zs["THR"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["OG1", "CB", "CA", "N"],  # torsion
    ["CG2", "CB", "CA", "OG1"],  # improper
    ["HB", "CB", "CA", "OG1"],  # improper
    ["HG1", "OG1", "CB", "CA"],  # torsion
    ["HG21", "CG2", "CB", "CA"],  # methyl torsion
    ["HG22", "CG2", "CB", "HG21"],  # improper
    ["HG23", "CG2", "CB", "HG21"],  # improper
]

basis_Zs["VAL"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG1", "CB", "CA", "N"],  # torsion
    ["CG2", "CB", "CA", "CG1"],  # improper
    ["HB", "CB", "CA", "CG1"],  # improper
    ["HG11", "CG1", "CB", "CA"],  # methyl torsion
    ["HG12", "CG1", "CB", "HG11"],  # improper
    ["HG13", "CG1", "CB", "HG11"],  # improper
    ["HG21", "CG2", "CB", "CA"],  # methyl torsion
    ["HG22", "CG2", "CB", "HG21"],  # improper
    ["HG23", "CG2", "CB", "HG21"],  # improper
]

basis_Zs["PHE"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "N", "C"],
    ["CB", "CA", "N", "C"],  # ... improper
    ["CG", "CB", "CA", "N"],  # torsion
    ["CD1", "CG", "CB", "CA"],  # torsion
    ["CD2", "CG", "CB", "CD1"],  # improper
    ["CE1", "CD1", "CG", "CB"],  # torsion but rigid
    ["CE2", "CD2", "CG", "CD1"],  # torsion but rigid
    ["CZ", "CE1", "CE2", "CD1"],  # improper
    ["HB2", "CB", "CA", "CG"],  # improper
    ["HB3", "CB", "CA", "CG"],  # improper
    ["HD1", "CD1", "CG", "CE1"],  # improper
    ["HD2", "CD2", "CG", "CE2"],  # improper
    ["HE1", "CE1", "CD1", "CZ"],  # improper
    ["HE2", "CE2", "CD2", "CZ"],  # improper
    ["HZ", "CZ", "CE1", "CE2"],  # improper
]

basis_Zs["MISC"] = [["H3", "N", "CA", "C"], ["OXT", "C", "CA", "N"]]


# Indexing is zero-based. end_res is exclusive.
MoleculeExtent = namedtuple("MoleculeExtent", "start_res end_res is_protein")


def mdtraj_to_z(topology, cart_ind, molecules=None, extra_basis=None):
    """
        topology: MDTraj topology
        cartesian: if not MDTraj selection string of atoms not to represent with internal coordinates
        molecules: list of `MoleculeExtent`s.
        extra_basis: a dictionary with res_names as key and lists of tuples as values
    """
    Z = []

    if molecules is None:
        molecules = [MoleculeExtent(0, topology.n_residues, True)]

    basis = basis_Zs.copy()
    if extra_basis:
        basis = {**basis, **extra_basis}

    residues = list(topology.residues)
    print("residues: ", residues)
    print("molecules: ", molecules)
    for molecule in molecules:
        for res_index in range(molecule.start_res, molecule.end_res):
            residue = residues[res_index]
            print("residue: ", residue)
            is_nterm = (res_index == molecule.start_res) and molecule.is_protein
            is_cterm = (res_index == molecule.end_res - 1) and molecule.is_protein
            print("is_nterm: ", is_nterm, " is c_term: ", is_cterm)

            if is_nterm:
                H1 = topology.select(f"resid {res_index} and name H1")
                if H1:
                    print("Renamed N-terminal H1 to H.")
                    topology.atom(H1[0]).name = "H"

            res_atoms = {atom.name: atom.index for atom in residue.atoms}
            print("res_atoms: ", res_atoms)
            res_name = residue.name
            print("res_name: ", res_name)
            res_basis = {e[0]: (e[1], e[2], e[3]) for e in basis[res_name]}
            # Add in extra atoms for N-terminus
            if is_nterm and molecule.is_protein:
                res_basis["H2"] = ("N", "CA", "H")
                res_basis["H3"] = ("N", "CA", "H2")
            # Add in extra atom for C-terminus
            if is_cterm and molecule.is_protein:
                res_basis["OXT"] = ("C", "CA", "O")

            for atom_name in res_atoms:
                atom_ind = res_atoms[atom_name]
                if atom_ind in cart_ind:
                    continue
                else:
                    try:
                        basis_atoms = res_basis[atom_name]
                    except KeyError as e:
                        raise KeyError(
                            f"Could not find atom {atom_name} in basis for {res_name}{res_index}."
                        ) from e
                    Z.append((atom_ind, [res_atoms[e] for e in basis_atoms]))

    if topology.n_atoms != len(Z) + len(cart_ind):
        print(len(Z))
        print(len(cart_ind))
        print(topology.n_atoms)
        print(Z)
        raise RuntimeError()

    return Z

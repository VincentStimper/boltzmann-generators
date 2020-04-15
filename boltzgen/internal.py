import torch
import torch.nn as nn
import math

class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


def calc_bonds(ind1, ind2, coords):
    """Calculate bond lengths

    Parameters
    ----------
    ind1 : torch.LongTensor
        A n_bond x 3 tensor of indices for the coordinates of particle 1
    ind2 : torch.LongTensor
        A n_bond x 3 tensor of indices for the coordinates of particle 2
    coords : torch.tensor
        A n_batch x n_coord tensor of flattened input coordinates
    """
    p1 = coords[:, ind1]
    p2 = coords[:, ind2]
    return torch.norm(p2 - p1, dim=2)


def calc_angles(ind1, ind2, ind3, coords):
    b = coords[:, ind1]
    c = coords[:, ind2]
    d = coords[:, ind3]
    bc = b - c
    bc = bc / torch.norm(bc, dim=2, keepdim=True)
    cd = d - c
    cd = cd / torch.norm(cd, dim=2, keepdim=True)
    cos_angle = torch.sum(bc * cd, dim=2)
    angle = torch.acos(cos_angle)
    return angle


def calc_dihedrals(ind1, ind2, ind3, ind4, coords):
    a = coords[:, ind1]
    b = coords[:, ind2]
    c = coords[:, ind3]
    d = coords[:, ind4]

    b0 = a - b
    b1 = c - b
    b1 = b1 / torch.norm(b1, dim=2, keepdim=True)
    b2 = d - c

    v = b0 - torch.sum(b0 * b1, dim=2, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=2, keepdim=True) * b1
    x = torch.sum(v * w, dim=2)
    b1xv = torch.cross(b1, v, dim=2)
    y = torch.sum(b1xv * w, dim=2)
    angle = torch.atan2(y, x)
    return -angle


def reconstruct_cart(cart, ref_atoms, bonds, angles, dihs):
    # Get the positions of the 4 reconstructing atoms
    p1 = cart[:, ref_atoms[:, 0], :]
    p2 = cart[:, ref_atoms[:, 1], :]
    p3 = cart[:, ref_atoms[:, 2], :]

    bonds = bonds.unsqueeze(2)
    angles = angles.unsqueeze(2)
    dihs = dihs.unsqueeze(2)

    # Compute the log jacobian determinant.
    jac = torch.sum(
        2 * torch.log(torch.abs(bonds.squeeze(2)))
        + torch.log(torch.abs(torch.sin(angles.squeeze(2)))),
        dim=1,
    )

    # Reconstruct the position of p4
    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2, dim=2)
    n = n / torch.norm(n, dim=2, keepdim=True)
    nn = torch.cross(v1, n, dim=2)
    nn = nn / torch.norm(nn, dim=2, keepdim=True)

    n = n * torch.sin(dihs)
    nn = nn * torch.cos(dihs)

    v3 = n + nn
    v3 = v3 / torch.norm(v3, dim=2, keepdim=True)
    v3 = v3 * bonds * torch.sin(angles)

    v1 = v1 / torch.norm(v1, dim=2, keepdim=True)
    v1 = v1 * bonds * torch.cos(angles)

    # Store the final position in x
    new_cart = p1 + v3 - v1

    return new_cart, jac


class InternalCoordinateTransform(Transform):
    def __init__(self, dims, z_indices=None, cart_indices=None, training_data=None):
        super().__init__()
        self.dims = dims
        with torch.no_grad():
            # Setup indexing.
            self._setup_indices(z_indices, cart_indices)
            self._validate_training_data(training_data)
            # Setup the mean and standard deviations for each internal coordinate.
            transformed, _ = self._fwd(training_data)
            self._setup_mean_bonds(transformed)
            transformed[:, self.bond_indices] -= self.mean_bonds
            self._setup_std_bonds(transformed)
            transformed[:, self.bond_indices] /= self.std_bonds
            self._setup_mean_angles(transformed)
            transformed[:, self.angle_indices] -= self.mean_angles
            self._setup_std_angles(transformed)
            transformed[:, self.angle_indices] /= self.std_angles
            self._setup_mean_dih(transformed)
            transformed[:, self.dih_indices] -= self.mean_dih
            self._fix_dih(transformed)
            self._setup_std_dih(transformed)
            transformed[:, self.dih_indices] /= self.std_dih
            scale_jac = -(
                torch.sum(torch.log(self.std_bonds))
                + torch.sum(torch.log(self.std_angles))
                + torch.sum(torch.log(self.std_dih))
            )
            self.register_buffer("scale_jac", scale_jac)

    def forward(self, x, context=None):
        trans, jac = self._fwd(x)
        trans[:, self.bond_indices] -= self.mean_bonds
        trans[:, self.bond_indices] /= self.std_bonds
        trans[:, self.angle_indices] -= self.mean_angles
        trans[:, self.angle_indices] /= self.std_angles
        trans[:, self.dih_indices] -= self.mean_dih
        self._fix_dih(trans)
        trans[:, self.dih_indices] /= self.std_dih
        return trans, jac + self.scale_jac

    def _fwd(self, x):
        x = x.clone()
        # we can do everything in parallel...
        inds1 = self.inds_for_atom[self.rev_z_indices[:, 1]]
        inds2 = self.inds_for_atom[self.rev_z_indices[:, 2]]
        inds3 = self.inds_for_atom[self.rev_z_indices[:, 3]]
        inds4 = self.inds_for_atom[self.rev_z_indices[:, 0]]

        # Calculate the bonds, angles, and torions for a batch.
        bonds = calc_bonds(inds1, inds4, coords=x)
        angles = calc_angles(inds2, inds1, inds4, coords=x)
        dihedrals = calc_dihedrals(inds3, inds2, inds1, inds4, coords=x)

        jac = -torch.sum(
            2 * torch.log(bonds) + torch.log(torch.abs(torch.sin(angles))), dim=1
        )

        # Replace the cartesian coordinates with internal coordinates.
        x[:, inds4[:, 0]] = bonds
        x[:, inds4[:, 1]] = angles
        x[:, inds4[:, 2]] = dihedrals
        return x, jac

    def inverse(self, x, context=None):
        # Gather all of the atoms represented as cartesisan coordinates.
        n_batch = x.shape[0]
        cart = x[:, self.init_cart_indices].view(n_batch, -1, 3)

        # Setup the log abs det jacobian
        jac = x.new_zeros(x.shape[0])

        # Loop over all of the blocks, where all of the atoms in each block
        # can be built in parallel because they only depend on atoms that
        # are already cartesian. `atoms_to_build` lists the `n` atoms
        # that can be built as a batch, where the indexing refers to the
        # original atom order. `ref_atoms` has size n x 3, where the indexing
        # refers to the position in `cart`, rather than the original order.
        for block in self.rev_blocks:
            atoms_to_build = block[:, 0]
            ref_atoms = block[:, 1:]

            # Get all of the bonds by retrieving the appropriate columns and
            # un-normalizing.
            bonds = (
                x[:, 3 * atoms_to_build]
                * self.std_bonds[self.atom_to_stats[atoms_to_build]]
                + self.mean_bonds[self.atom_to_stats[atoms_to_build]]
            )
            # Get all of the angles by retrieving the appropriate columns and
            # un-normalizing.
            angles = (
                x[:, 3 * atoms_to_build + 1]
                * self.std_angles[self.atom_to_stats[atoms_to_build]]
                + self.mean_angles[self.atom_to_stats[atoms_to_build]]
            )
            # Get all of the dihedrals by retrieving the appropriate columns and
            # un-normalizing.
            dihs = (
                x[:, 3 * atoms_to_build + 2]
                * self.std_dih[self.atom_to_stats[atoms_to_build]]
                + self.mean_dih[self.atom_to_stats[atoms_to_build]]
            )
            # Fix the dihedrals to lie in [-pi, pi].
            dihs = torch.where(dihs < math.pi, dihs + 2 * math.pi, dihs)
            dihs = torch.where(dihs > math.pi, dihs - 2 * math.pi, dihs)

            # Compute the cartesian coordinates for the newly placed atoms.
            new_cart, cart_jac = reconstruct_cart(cart, ref_atoms, bonds, angles, dihs)
            jac = jac + cart_jac

            # Concatenate the cartesian coordinates for the newly placed
            # atoms onto the full set of cartesian coordiantes.
            cart = torch.cat([cart, new_cart], dim=1)
        # Permute cart back into the original order and flatten.
        cart = cart[:, self.rev_perm_inv]
        cart = cart.view(n_batch, -1)
        return cart, jac - self.scale_jac

    def _setup_mean_bonds(self, x):
        mean_bonds = torch.mean(x[:, self.bond_indices], dim=0)
        self.register_buffer("mean_bonds", mean_bonds)

    def _setup_std_bonds(self, x):
        # Adding 1e-4 is for numerical stability but results in some
        # dimensions being not properly normalised e.g. bond lengths
        # which can have stds of the order 1e-7
        # The flow will then have to fit to a very concentrated dist
        std_bonds = torch.std(x[:, self.bond_indices], dim=0) #+ 1e-4
        self.register_buffer("std_bonds", std_bonds)

    def _setup_mean_angles(self, x):
        mean_angles = torch.mean(x[:, self.angle_indices], dim=0)
        self.register_buffer("mean_angles", mean_angles)

    def _setup_std_angles(self, x):
        std_angles = torch.std(x[:, self.angle_indices], dim=0) #+ 1e-4
        self.register_buffer("std_angles", std_angles)

    def _setup_mean_dih(self, x):
        sin = torch.mean(torch.sin(x[:, self.dih_indices]), dim=0)
        cos = torch.mean(torch.cos(x[:, self.dih_indices]), dim=0)
        mean_dih = torch.atan2(sin, cos)
        self.register_buffer("mean_dih", mean_dih)

    def _fix_dih(self, x):
        dih = x[:, self.dih_indices]
        dih = torch.where(dih < -math.pi, dih + 2 * math.pi, dih)
        dih = torch.where(dih > math.pi, dih - 2 * math.pi, dih)
        x[:, self.dih_indices] = dih

    def _setup_std_dih(self, x):
        std_dih = torch.std(x[:, self.dih_indices], dim=0)
        std_dih = torch.ones_like(std_dih)
        self.register_buffer("std_dih", std_dih)

    def _validate_training_data(self, training_data):
        if training_data is None:
            raise ValueError(
                "InternalCoordinateTransform must be supplied with training_data."
            )

        if len(training_data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_samp = training_data.shape[0]
        n_dim = training_data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

        if not n_samp >= 1:
            raise ValueError("training_data must have n_samp > 1.")

    def _setup_indices(self, z_indices, cart_indices):
        n_atoms = self.dims // 3
        ind_for_atom = torch.zeros(n_atoms, 3, dtype=torch.long)
        for i in range(n_atoms):
            ind_for_atom[i, 0] = 3 * i
            ind_for_atom[i, 1] = 3 * i + 1
            ind_for_atom[i, 2] = 3 * i + 2
        self.register_buffer("inds_for_atom", ind_for_atom)

        sorted_z_indices = topological_sort(z_indices)
        sorted_z_indices = [
            [item[0], item[1][0], item[1][1], item[1][2]] for item in sorted_z_indices
        ]
        rev_z_indices = list(reversed(sorted_z_indices))

        mod = [item[0] for item in sorted_z_indices]
        modified_indices = []
        for index in mod:
            modified_indices.extend(self.inds_for_atom[index])
        bond_indices = list(modified_indices[0::3])
        angle_indices = list(modified_indices[1::3])
        dih_indices = list(modified_indices[2::3])

        self.register_buffer("modified_indices", torch.LongTensor(modified_indices))
        self.register_buffer("bond_indices", torch.LongTensor(bond_indices))
        self.register_buffer("angle_indices", torch.LongTensor(angle_indices))
        self.register_buffer("dih_indices", torch.LongTensor(dih_indices))
        self.register_buffer("sorted_z_indices", torch.LongTensor(sorted_z_indices))
        self.register_buffer("rev_z_indices", torch.LongTensor(rev_z_indices))

        #
        # Setup indexing for reverse pass.
        #
        # First, create an array that maps from an atom index into mean_bonds, std_bonds, etc.
        atom_to_stats = torch.zeros(n_atoms, dtype=torch.long)
        for i, j in enumerate(mod):
            atom_to_stats[j] = i
        self.register_buffer("atom_to_stats", atom_to_stats)

        # Next create permutation vector that is used in the reverse pass. This maps
        # from the original atom indexing to the order that the cartesian coordinates
        # will be built in. This will be filled in as we go.
        rev_perm = torch.zeros(n_atoms, dtype=torch.long)
        self.register_buffer("rev_perm", rev_perm)
        # Next create the inverse of rev_perm. This will be filled in as we go.
        rev_perm_inv = torch.zeros(n_atoms, dtype=torch.long)
        self.register_buffer("rev_perm_inv", rev_perm_inv)

        # Create the list of columns that form our initial cartesian coordintes.
        init_cart_indices = self.inds_for_atom[cart_indices].view(-1)
        self.register_buffer("init_cart_indices", init_cart_indices)

        # Update our permutation vectors for the initial cartesian atoms.
        for i, j in enumerate(cart_indices):
            self.rev_perm[i] = torch.as_tensor(j, dtype=torch.long)
            self.rev_perm_inv[j] = torch.as_tensor(i, dtype=torch.long)

        # Break Z into blocks, where all of the atoms within a block can be built
        # in parallel, because they only depend on already-cartesian atoms.
        all_cart = set(cart_indices)
        current_cart_ind = i + 1
        blocks = []
        while sorted_z_indices:
            next_z_indices = []
            next_cart = set()
            block = []
            for atom1, atom2, atom3, atom4 in sorted_z_indices:
                if (atom2 in all_cart) and (atom3 in all_cart) and (atom4 in all_cart):
                    # We can build this atom from existing cartesian atoms, so we add
                    # it to the list of cartesian atoms available for the next block.
                    next_cart.add(atom1)

                    # Add this atom to our permutation marices.
                    self.rev_perm[current_cart_ind] = atom1
                    self.rev_perm_inv[atom1] = current_cart_ind
                    current_cart_ind += 1

                    # Next, we convert the indices for atoms2-4 from their normal values
                    # to the appropriate indices to index into the cartesian array.
                    atom2_mod = self.rev_perm_inv[atom2]
                    atom3_mod = self.rev_perm_inv[atom3]
                    atom4_mod = self.rev_perm_inv[atom4]

                    # Finally, we append this information to the current block.

                    block.append([atom1, atom2_mod, atom3_mod, atom4_mod])
                else:
                    # We can't build this atom from existing cartesian atoms,
                    # so put it on the list for next time.
                    next_z_indices.append([atom1, atom2, atom3, atom4])
            sorted_z_indices = next_z_indices
            all_cart = all_cart.union(next_cart)
            block = torch.as_tensor(block, dtype=torch.long)
            blocks.append(block)
        self.rev_blocks = blocks


def topological_sort(graph_unsorted):
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)

    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occured.")

    return graph_sorted

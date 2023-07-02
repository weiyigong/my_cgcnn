import csv
import functools
import json
import os.path as osp
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, Dataset

warnings.filterwarnings("ignore")


#################################### copied from CGCNN
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


#################################### copied from CGCNN


class MyDataset(Dataset):
    def __init__(self, root_dir, radius=8.0, step=0.2):
        super().__init__()
        self.root_dir = root_dir
        self.radius = radius

        with open(osp.join(root_dir, 'id_prop.csv'), 'r') as f:
            reader = csv.reader(f)
            self.raw_data = [row for row in reader]

        self.ari = AtomCustomJSONInitializer(osp.join(root_dir, 'atom_init.json'))
        self.gdf = GaussianDistance(dmin=0., dmax=self.radius, step=step)

    def __len__(self):
        return len(self.raw_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, target = self.raw_data[idx]
        struc = Structure.from_file(osp.join(self.root_dir, cif_id + '.cif'))
        ctr_idx, nbr_idx, _, dis = struc.get_neighbor_list(self.radius)

        x = np.vstack([self.ari.get_atom_fea(site.specie.number) for site in struc])
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(np.vstack((ctr_idx, nbr_idx)), dtype=torch.long)
        edge_attr = torch.tensor(self.gdf.expand(dis), dtype=torch.float)
        target = torch.tensor([float(target)], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target)

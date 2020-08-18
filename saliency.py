import os
import numpy as np

import itertools
import pickle

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


class VanillaGrad(object):
    def __init__(self, th_max=float('-inf'), th_min=float('inf')):
        super(VanillaGrad, self).__init__()
        self.vmax = th_max
        self.vmin = th_min

    def calc_range(self, saliency):
        for v in saliency:
            self.vmax = max(self.vmax, np.max(v))
            self.vmin = min(self.vmin, np.min(v))

        return self.vmin, self.vmax

    def get_scaler(self, v_range):
        def scaler(saliency_):
            saliency = np.copy(saliency_)
            minv, maxv = v_range
            if maxv == minv:
                saliency = np.zeros_like(saliency)
            else:
                pos = saliency >= 0.0
                saliency[pos] = saliency[pos]/maxv
                nega = saliency < 0.0
                saliency[nega] = saliency[nega]/(np.abs(minv))
            for idx in range(len(saliency)):
                if saliency[idx] > 1:
                    saliency[idx] = 1
                elif saliency[idx] < -1:
                    saliency[idx] = -1

            return saliency
        return scaler

    def color_fn(self, x):
        if x > 0:
            # Red for positive value
            return 1., 1. - x, 1. - x
        else:
            # Blue for negative value
            x *= -1
            return 1. - x, 1. - x, 1.

    def is_visible(self, begin, end):
        if begin <= 0 or end <= 0:
            return 0
        elif begin >= 1 or end >= 1:
            return 1
        else:
            return (begin + end) * 0.5


    def save_saliency_map(self, args, n_atom_list, saliency_list, pdb_list):
        cliped_list = []
        for idx, saliency_item in enumerate(saliency_list):
            results = np.stack(saliency_item, axis=0)
            h = np.sum(results, axis=2)
            for i in range(len(h)):
                cliped_list.append(h[i, :n_atom_list[idx][i]])

        saliency_vanilla = np.asarray(cliped_list)
        v_range_vanilla = self.calc_range(saliency_vanilla)
        scaler_vanilla = self.get_scaler(v_range_vanilla)

        for pdb_code, vanilla in zip(pdb_list, cliped_list):

            pocket_fname = args.data_fpath + '/' + pdb_code + '/' + pdb_code + '_pocket.pdb'
            for f in os.listdir(args.data_fpath + '/' + pdb_code):
                if f.endswith('.sdf'):
                    ligand_name = f[:3]
                    ligand_fname = args.data_fpath + '/' + pdb_code + '/' + ligand_name + '.sdf'
                    break
            for m1 in Chem.SDMolSupplier(ligand_fname): break
            m2 = Chem.MolFromPDBFile(pocket_fname)

            #with open(args.data_fpath + '{0}/{0}.pair_{1}.pkl'.format(pdb_code, args.distance), 'rb') as f:
            #    m1, m2 = pickle.load(f)
            ligand = m1
            l_num_atoms = ligand.GetNumAtoms()
            rdDepictor.Compute2DCoords(ligand)
            Chem.SanitizeMol(ligand)
            Chem.Kekulize(ligand)
            l_n_atoms = ligand.GetNumAtoms()

            pocket = m2
            p_num_atoms = pocket.GetNumAtoms()
            rdDepictor.Compute2DCoords(pocket)
            Chem.SanitizeMol(pocket)
            Chem.Kekulize(pocket)
            p_n_atoms = pocket.GetNumAtoms()

            l_saliency = vanilla[:l_n_atoms]
            p_saliency = vanilla[l_n_atoms:]

            l_saliency = scaler_vanilla(l_saliency)
            threshold = np.min(l_saliency)

            highlight_atoms = list(map(lambda g: g.__int__(), np.where(l_saliency >= threshold)[0]))
            atom_colors = {i: self.color_fn(e) for i, e in enumerate(l_saliency)}
            bondlist = [bond.GetIdx() for bond in ligand.GetBonds()]

            def color_bond(bond):
                begin = l_saliency[bond.GetBeginAtomIdx()]
                end = l_saliency[bond.GetEndAtomIdx()]
                return self.color_fn(self.is_visible(begin, end))

            bondcolorlist = {i: color_bond(bond) for i, bond in enumerate(ligand.GetBonds())}
            drawer = rdMolDraw2D.MolDraw2DSVG(500, 375)
            drawer.DrawMolecule(
                ligand, highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors, highlightBonds=bondlist,
                highlightBondColors=bondcolorlist)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            ext = '.png'

            if not os.path.isdir(args.map_path):
                os.mkdir(args.map_path)
            save_filepath = os.path.join(args.map_path, pdb_code + ext)

            if save_filepath:
                extention = save_filepath.split('.')[-1]
                if extention == 'svg':
                    print('saving svg to {}'.format(save_filepath))
                    with open(save_filepath, 'w') as f:
                        f.write(svg)
                elif extention == 'png':
                    import cairosvg

                    print('saving png to {}'.format(save_filepath))
                    cairosvg.svg2png(bytestring=svg, write_to=save_filepath)
                else:
                    raise ValueError(
                        'Unsupported extention {} for save_filepath {}'.format(extention, save_filepath))




import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import pyplot as plt
import os, random, glob
from IPython.display import SVG, HTML

def get_kinase_set():
    kinase = set()
    for line in open('keys/kinase_list', 'rt'):
        line = line.strip()
        kinase.add(line)
    return kinase

def read_keyfile(fname, local=False):
    keys = []
    for line in open(fname):
        line = line.rstrip()
        if line.startswith('#'):
            continue
        it = line.split('\t')
        if local:
            pdb_code, value = it[0], float(it[1])
        else:
            pdb_code, ligand_name, year, value = it[0], it[1], int(it[2]), float(it[3])
        if os.path.exists(f'cbidata/{pdb_code}'):
            keys.append((pdb_code, value))
    return keys

def read_ligand(dirname):
    sdf_found = False
    for f in os.listdir(dirname):
        if f.endswith('.sdf'):
            sdf_found = True
            sdf_fname = f'{dirname}/{f}'
            for ligand_mol in Chem.SDMolSupplier(sdf_fname):
                break
            break
    return ligand_mol if sdf_found else None

def filter_and_stratify(*dataset, random_stratify=False, kinase_check=False):
    kinase = get_kinase_set()
    all_data = []
    for _ in dataset:
        all_data += _
    if random_stratify:
        random.shuffle(all_data)
    else:
        data = sorted(all_data, key=lambda x: x[1], reverse=True)
    train = []
    test = []
    test2 = []

    count = 0
    for key, pkd in all_data:
        if kinase_check and key not in kinase:
            continue

        if pkd < 2 or 11 < pkd:
            test2.append((key, pkd))
            continue

        ligand_mol = read_ligand(f'cbidata/{key}')

        n_atoms = ligand_mol.GetNumAtoms()
        if n_atoms < 10 or 45 < n_atoms:
            test2.append((key, pkd))
            continue
        if QED.qed(ligand_mol) < 0.4:
            test2.append((key, pkd))
            continue

        count += 1

        if count % 5 == 0:
            test.append((key, pkd))
        else:
            train.append((key, pkd))
    return train, test, test2

def write_keys(keys, oname):
    with open(oname, 'wt') as out:
        for key, pkd in keys:
            print(key, pkd, sep='\t', file=out)

def molsvg(mol, width=150, height=100):
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    view = rdMolDraw2D.MolDraw2DSVG(width, height)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    option = view.drawOptions()
    option.circleAtoms=False
    view.DrawMolecule(tm)
    view.FinishDrawing()
    svg = view.GetDrawingText()
    return SVG(svg)

def myplot(y_obs, y_pred, Los, Lps):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    xs = list(range(len(Los)))
    ax1.plot(xs, Los)
    ax1.plot(xs, Lps)
    if min(Los) < 10 and min(Lps) < 10:
        _ymax = 15
    else:
        _ymax = 60
    ax1.set_ylim(0, _ymax)
    ax2.scatter(y_obs, y_pred)
    ax2.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
    ax2.set_xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    ax2.set_ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    ax2.set_xlabel('Experimental', fontsize=16)
    ax2.set_ylabel('Calculated', fontsize=16)
    ax2.tick_params(labelsize=16)
    plt.subplots_adjust(wspace=.5)
    plt.show()

def write_results_to_csv(L, X, y, err, epoch):
    df = pd.DataFrame(dict(PDB=L, pval=X, predicted=y, err=err))
    df.to_csv(f'results_{epoch}.tsv', sep='\t')
    return df

def show_bad_molecules(L, X, y, err, N):
    bads = list(filter(lambda v: 2.5 <= v[3], sorted(zip(L, X, y, X-y), key=lambda v: v[2], reverse=True)))
    if N//5 < len(bads):
        print(f'--- Too many off-valued molecules ({len(bads)}/{len(L)}) ---')
        print()
        return
    elif len(bads) == 0:
        return
    svgs = []
    for bad in bads:
        pdb_code = bad[0]
        sdf_name = glob.glob(f'cbidata/{pdb_code}/*.sdf')[0]
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromMolFile(sdf_name)))
        svgs.append(molsvg(mol, width=250, height=150))
    html = '<table border="1" style="font-size: 120%;">'
    for svg, (pdb_code, x, y, err) in zip(svgs, bads):
        html += f'<tr><td>{svg.data}</td><td>{pdb_code}</td><td>{x:.3f}</td><td>{y:.3f}</td><td>{err:.3f}</td></tr>'
    html += '</table>'
    display(HTML(html))

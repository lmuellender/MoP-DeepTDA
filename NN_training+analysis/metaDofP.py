import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlcvs.utils import load_dataframe
from matplotlib.animation import FuncAnimation
import os
from itertools import combinations, product
from plots import plot_cv_ramachandran
from tqdm import tqdm


def convert_colvar(path, dest, write_biasfile=True):
    """ Covert Colvar files from a metaD of paths simulation to Colvars of the actual paths. """
    if not os.path.exists(dest):
        os.mkdir(dest)
    
    files = sorted(os.listdir(path), key=lambda x: int(x.split('.')[-1]))
    n_beads = len(files)
    tst_colvar = load_dataframe(path+files[0])
    header = tst_colvar.columns
    n_paths = len(tst_colvar) - 1
    data = np.zeros((n_beads, n_paths, len(header)))
    print('Loading metaD of Paths Colvars...')
    min_l = n_paths
    for i,f in tqdm(enumerate(files), total=len(files)):
        try:
            df = load_dataframe(path+f)
        except Exception:
            print('ERROR: failed to load trajectory', i, f)
        l = len(df) - 1 
        if l < n_paths:
            print('WARING: incomplete trajectory {} of length {} instead of {}'.format(f,l,n_paths))
            if l < min_l:
                min_l = l
        data[i,:l] = df.values[:-1]

    if min_l < n_paths:
        print('WARNING: truncating all data to fit minimum length', min_l)
        data = data[:,:min_l]
    data = data.transpose(1, 0, 2)

    if 'opes.bias' in header and write_biasfile:
        opes = df[['time', 'opes.bias', 'opes.rct']]
        fn = '../' + dest.split('/')[-2] + '_opes.dat'
        opes.to_csv(dest + fn)
    
    print('Writing path Colvars...')
    pddata = []
    for i, traj in tqdm(enumerate(data), total=len(data)):
        colvar = pd.DataFrame(traj, columns=header)
        colvar.to_csv(dest+'path.'+str(i+1), sep=' ')
        colvar['path_id'] = i
        pddata.append(colvar)
    return pd.concat(pddata, ignore_index=True)


def read_lammps_dump(filepath):
    ### Adapted from Adam Plowman (https://github.com/aplowman/lammps-parse/). ###
    """Parse a Lammps dump file.
    Parameters
    ----------
    path : str or Path
        File path to the Lammps log file to be read.
    Returns
    -------
    dump_data : dict
        Dict with the following keys:
            time_step : int 
            num_atoms : int 
            box_tilt : bool
            box_periodicity : list of str
            box : ndarray
            supercell : ndarray
            atom_sites : ndarray
            atom_types : ndarray
            atom_pot_energy : ndarray
            atom_disp : ndarray
            vor_vols : ndarray
            vor_faces: ndarray
    Notes
    -----
    This is not generalised, in terms of the fields present in the `ATOMS` block, but
    could be developed to be more-generalised in the future.
    """

    # Search strings
    TS = 'ITEM: TIMESTEP'
    NUM_ATOMS = 'ITEM: NUMBER OF ATOMS'
    BOX = 'ITEM: BOX BOUNDS'
    ATOMS = 'ITEM: ATOMS'
    TILT_FACTORS = 'xy xz yz'

    ts_all = []
    num_atoms = None
    box_tilt = False
    box_periodicity = None
    box = []
    atom_sites_all = []
    atom_types = None
    atom_ids = None

    with open(filepath, 'r', encoding='utf-8', newline='') as df:
        mode = 'scan'
        for ln in df:

            ln = ln.strip()
            ln_s = ln.split()

            if TS in ln:
                mode = 'ts'
                continue

            elif NUM_ATOMS in ln:
                mode = 'num_atoms'
                continue

            elif BOX in ln:
                mode = 'box'
                box_ln_idx = 0
                box_periodicity = [ln_s[-i] for i in [3, 2, 1]]
                if TILT_FACTORS in ln:
                    box_tilt = True
                continue

            elif ATOMS in ln:
                mode = 'atoms'
                headers = ln_s[2:]

                x_col = headers.index('x')
                y_col = headers.index('y')
                z_col = headers.index('z')

                atom_mol_col = headers.index('mol')
                atom_type_col = headers.index('type')
                atom_id_col = headers.index('id')

                atom_ln_idx = 0
                atom_sites = np.zeros((3, num_atoms))
                atom_mols = np.zeros((num_atoms,), dtype=int)
                atom_types = np.zeros((num_atoms,), dtype=int)
                atom_ids = np.zeros((num_atoms,), dtype=int)

                continue

            if mode == 'ts':
                ts = int(ln)
                mode = 'scan'

            elif mode == 'num_atoms':
                num_atoms = int(ln)
                mode = 'scan'

            elif mode == 'box':
                box.append([float(i) for i in ln_s])
                box_ln_idx += 1
                if box_ln_idx == 3:
                    mode = 'scan'

            elif mode == 'atoms':

                atom_sites[:, atom_ln_idx] = [
                    float(i) for i in (ln_s[x_col], ln_s[y_col], ln_s[z_col])]

                atom_mols[atom_ln_idx] = int(ln_s[atom_mol_col])
                atom_types[atom_ln_idx] = int(ln_s[atom_type_col])
                atom_ids[atom_ln_idx] = int(ln_s[atom_id_col])

                atom_ln_idx += 1
                if atom_ln_idx == num_atoms:
                    mode = 'scan'
                    # new block incoming
                    ts_all.append(ts)
                    atom_sites_all.append(atom_sites)

    ts_all = np.array(ts_all)
    atom_sites_all = np.array(atom_sites_all)
    # !!! account for difference in units between lammps (Ang) and plumed (nm)
    atom_sites_all /= 10.
    # i.e. coords are now in nm

    dump_data = {
        'time_step': ts_all,
        'num_atoms': num_atoms,
        'box_tilt': box_tilt,
        'box_periodicity': box_periodicity,
        'box': box,
        # 'supercell': supercell,
        'atom_sites': atom_sites_all,
        'atom_mols': atom_mols,
        'atom_types': atom_types,
        'atom_ids': atom_ids
    }

    return dump_data


def read_paths(path, fnames):
    data = []
    print('Reading paths...')
    for i,fn in tqdm(enumerate(fnames), total=len(fnames)):
        p = path+fn
        if os.path.exists(p):
            # df = utils.load_dataframe(p)
            try:
                df = pd.read_csv(p, delimiter=' ', comment='#', index_col=0)
            except:
                pass
            df['path_id'] = int(fn.split('.')[-1])
            if not df.empty:
                data.append(df)
    return pd.concat(data, ignore_index=True)


def mdopdump2pdb(path, path_id, pdbfile, stride=20):
    # read single trajectory from MDoP LAMMPS dump files and convert to PDB
    # file name format traj.*.lammpstrj
    # path: path to directory containing LAMMPS dump files
    # path_id: id of path to convert to pdb
    # pdbfile: path to write PDB file
    fnames = np.array(sorted(os.listdir(path), key = lambda x: int(x.split('.')[-2])))
    n_beads = len(fnames)
    tst_dump = read_lammps_dump(path+fnames[0])
    n_paths = len(tst_dump['time_step'])
    # data = np.zeros((n_beads, n_paths, 5))
    data = []
    # take only every <stride>'th bead
    idx = np.linspace(0, n_beads-1, stride, dtype=int)
    print('Reading dump files...')
    for i,f in tqdm(enumerate(fnames[idx]), total=len(idx)):
        dump = read_lammps_dump(path+f)
        xyz = dump['atom_sites'][path_id]
        id = dump['atom_ids']
        tp = dump['atom_types']
        data.append((id, xyz, tp))
    # transpose so that 1st axis corresponds to paths
    # data = data.transpose(1,0,2)

    atom_types = ['H','C','C','O','N','H','H'] # LUT for atom types 1-7
    with open(pdbfile, 'w') as file:
        for conf in data:
            id = conf[0]
            xyz = conf[1]
            types = conf[2]
            for i,p,tp in zip(id,xyz.T,types):
                if atom_types[tp-1]=='H':
                    continue
                line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
                    'ATOM', i, atom_types[tp-1], '','ALA', '', 1, '', p[0], p[1], p[2], 1.0, 1.0
                )
                file.write(line)
            file.write('END\n')
        print('Reference path written to '+pdbfile)

    return data



def mic_distance(v1, v2, period):
    "returns distance vector (v1-v2)"
    images = np.array(list(product([-1,0,1], repeat=3)))
    d = np.array([v1 - v2 + i*period for i in images]).transpose(1,0,2)
    min_d = np.array([d[i,m] for i,m in enumerate(np.argmin(np.linalg.norm(d, axis=-1), axis=-1))])
    return min_d

def calculate_distance(dump_data, pairs):
    """ calculate distances from parsed lammps dump for given atom ids """
    ids = dump_data['atom_ids']
    xyz = dump_data['atom_sites']
    x, y, z = xyz[:,0,:], xyz[:,1,:], xyz[:,2,:]
    box = dump_data['box'][0]
    period = box[1] - box[0] # assumes square, centered simulation box
    # convert atom ids to indices
    i = np.array([ids.tolist().index(i) for i in pairs[:,0]])
    j = np.array([ids.tolist().index(j) for j in pairs[:,1]])
    dist = np.sqrt((x[:,i]-x[:,j])**2 + (y[:,i]-y[:,j])**2 + (z[:,i]-z[:,j])**2)
    dist = np.array([np.linalg.norm(mic_distance(a[:,i].T, a[:,j].T, period), axis=-1) for a in xyz])
    return dist


def calculate_dihedral(dump_data, atoms):
    """ calculate dihedral angle between 4 atoms in a lammps dump given atom ids """
    ids = dump_data['atom_ids']
    xyz = dump_data['atom_sites']
    box = dump_data['box'][0]
    period = box[1] - box[0] # assumes square, centered simulation box
    # convert atom ids to indices
    idx = np.array([ids.tolist().index(i) for i in atoms])
    p = xyz[:,:,idx].transpose(0,2,1)
    b = np.array([mic_distance(p[:,i+1], p[:,i], period) for i in range(len(atoms)-1)]).transpose(1,0,2)

    v1 = np.cross(b[:,0], b[:,1])
    v2 = np.cross(b[:,1], b[:,2])
    y = np.linalg.norm(b[:,1],axis=1) * np.array([i.dot(j) for i,j in zip(b[:,0], v2)]) 
    x = np.array([i.dot(j) for i,j in zip(v1, v2)])
    return np.arctan2(y,x)


def convert_trajectories(path, dest):
    """ convert MetaD of Paths trajectories into the physical paths with pairwise dist & dihedrals """
    files = os.listdir(path)

    heavy_atoms = [2,5,6,7,9,11,15,16,17,19]
    pairs = np.array(list(combinations(heavy_atoms, 2)))
    phi_ids = [5,7,9,15]
    psi_ids = [7,9,15,17]

    n_beads = len(files)
    tst_dump = read_lammps_dump(path+files[0])
    n_paths = len(tst_dump['time_step'])

    print("Loading path-polymer trajectories...")
    # 45 dist + 2 dihedrals + time
    data = np.zeros((n_beads, n_paths, 48))
    for f in files:
        bead_idx = int(f.split('.')[1])-1
        dump = read_lammps_dump(path+f)
        # bead index = physical time step
        data[bead_idx, :, 0] = float(bead_idx) 
        data[bead_idx, :, 1:46] = calculate_distance(dump, pairs)
        data[bead_idx, :, 46] = calculate_dihedral(dump, phi_ids)
        data[bead_idx, :, 47] = calculate_dihedral(dump, psi_ids)

    # transpose to save as path data
    data = data.transpose(1, 0, 2)

    print("Writing path data...")
    headers = ['time'] + ['d'+str(i+1) for i in range(45)] + ['phi', 'psi']
    for i,traj in enumerate(data):
        dataframe = pd.DataFrame(traj, columns=headers)
        fname = 'path.'+str(i)
        dataframe.to_csv(dest+fname, sep=' ')


def plot_trajectories(traj_data, title = '', xlabel='phi', ylabel='psi', freq=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(title)
    if freq:
        pp = ax.hexbin(data[xlabel], data[ylabel], gridsize=80, extent=(-np.pi,np.pi,-np.pi,np.pi), cmap='afmhot_r')
        plt.colorbar(pp, ax=ax, label='count')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.scatter(data[xlabel], data[ylabel], s=.5)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_ylabel(ylabel)

    # FES isolines
    # gridx, gridy, fes = np.loadtxt('reference_fes/fes.dat', skiprows=1, delimiter=' ', unpack=True)
    # gridx = gridx.reshape((int(np.sqrt(len(gridx))), -1))
    # gridy = gridy.reshape((int(np.sqrt(len(gridy))), -1))
    # fes = fes.reshape((int(np.sqrt(len(fes))), -1))
    # levels = np.arange(np.min(fes), 70, 10)
    # ax.contour(gridx, gridy, fes, levels=levels, linewidths=.5, alpha=.5, colors='black')

    return fig, ax


def animate_trajectories(traj_frames):
    # FES isolines
    gridx, gridy, fes = np.loadtxt('reference_fes/fes.dat', skiprows=1, delimiter=' ', unpack=True)
    gridx = gridx.reshape((int(np.sqrt(len(gridx))), -1))
    gridy = gridy.reshape((int(np.sqrt(len(gridy))), -1))
    fes = fes.reshape((int(np.sqrt(len(fes))), -1))
    levels = np.arange(np.min(fes), 70, 10)
    
    # animate
    fig, ax = plt.subplots()
    def ani_func(frame):
        ax.clear()
        ax.set_title('Path #'+str(frame))
        ax.set_xlabel('$\phi$')
        ax.set_ylabel('$\psi$')
        ax.contour(gridx, gridy, fes, levels=levels, linewidths=.5, alpha=.5, colors='black')
        ax.plot(traj_frames[frame]['phi'], traj_frames[frame]['psi'], marker='.', ms=1, linewidth=.7)

    ani = FuncAnimation(fig, ani_func, frames = np.arange(0, len(traj_frames), 1), interval=50)
    return fig, ax, ani



if __name__ == "__main__":

    folder = '/Volumes/mySSD/data/paper/alanine/mdop/'
    mdop_path = '/Volumes/Daten/thesis_data/metaDofP_N384_gmex_long_colvar/'
    ppath = '/Volumes/Daten/thesis_data/data/paths_gmex_long_N384/'
    # convert_colvar(mdop_path, ppath)

    # opes = pd.read_csv('/Volumes/Daten/thesis_data/data/paths_gmex_long_N384_opes.dat')
    # opes.plot('time', 'opes.rct')


    #---TEST---
    # colvar_path = '/Volumes/Daten/thesis_data/data/Colvar_test'
    dump_path = folder+'dump_N512_tda3st_tst/'
    dump = mdopdump2pdb(dump_path, 2103, folder+'ala_path.pdb')
    i=0
    # heavy_atoms = [2,5,6,7,9,11,15,16,17,19]
    # pairs = np.array(list(combinations(heavy_atoms, 2)))
    # dist = calculate_distance(dump, pairs)
    # phi = calculate_dihedral(dump, [5,7,9,15])
    # psi = calculate_dihedral(dump, [7,9,15,17])
    # dump_data = np.concatenate([dist, phi, psi], axis=0)
    # colvar = pd.read_csv(colvar_path, delimiter=' ', comment='#').values[:,2:]

    # print('Phi identical:', np.allclose(phi[1:],colvar[:,-2]))
    # print('Psi identical:', np.allclose(psi[1:], colvar[:,-1]))
    # print('distances identical:', np.allclose(dist[1:], colvar[:,:-2]))
    # #----------


    # load some paths
    files = sorted(os.listdir(ppath), key = lambda x: int(x.split('.')[-1]))
    data = []
    # for fn in np.random.choice(files[1000:]):
    for i,fn in enumerate(files[8000:9000]):
        df = pd.read_csv(ppath+fn, delimiter=' ', index_col=0)
        data.append(df)
    # pathA = '/Volumes/Daten/thesis_data/data/ColvarA'
    # pathB = '/Volumes/Daten/thesis_data/data/ColvarB'
    # data = [load_dataframe(pathA), load_dataframe(pathB)]

    # labels/keys for plotting / getting data
    lx = 'phi'
    ly = 'psi'

    plot_trajectories(data, freq=True, xlabel=lx, ylabel=ly)

    # plot_cv_ramachandran(pd.concat(data), cv='CV', draw_iso=True)

    gridx, gridy, fes = np.loadtxt('reference_fes/fes.dat', skiprows=1, delimiter=' ', unpack=True)
    gridx = gridx.reshape((int(np.sqrt(len(gridx))), -1))
    gridy = gridy.reshape((int(np.sqrt(len(gridy))), -1))
    fes = fes.reshape((int(np.sqrt(len(fes))), -1))
    levels = np.arange(np.min(fes), 70, 10)
    contour = plt.contour(gridx, gridy, fes, levels=levels, linewidths=.5, alpha=.5, colors='black')


    # animate
    fig, ax = plt.subplots()
    def ani_func(frame):
        ax.clear()
        ax.set_title('Path #'+str(frame))
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.contour(contour)
        ax.plot(data[frame][lx], data[frame][ly], marker='.', ms=1, linewidth=.7)

    ani = FuncAnimation(fig, ani_func, frames = np.arange(0, len(data), 1), interval=50)



    # polymer energy
    # pe_path = '../cluster_data/metaDofP/metaDofP_1/polymer/'
    # data = []
    # for i in range(432):
    #     data.append(np.loadtxt(pe_path+'polymer.'+str(i+1)+'.nrg', delimiter=' ', comments='#'))
    # data = np.array(data)
    # # data = np.loadtxt(pe_path+'polymer.100.nrg', delimiter=' ', skiprows=1)
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(432), data[:,4191,1]+data[:,4191,2])

    plt.show()

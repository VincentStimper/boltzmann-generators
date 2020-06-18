#%%
# Import packages
import torch
import torch.nn as nn
from openmmtools.constants import kB
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import AlanineDipeptideImplicit
import numpy as np
import sys
import normflow as nf
sys.path.append('../')
import boltzgen.openmm_interface as omi
import boltzgen.mixed as mixed
from boltzgen.distributions import Boltzmann
from boltzgen.flows import CoordinateTransform
import mdtraj
from matplotlib import pyplot as plt
from tqdm import tqdm
from boltzgen.utils import KSD, blockKSD, get_median_estimate

# Set up simulation object
temperature = 1000
kT = kB * temperature

testsystem = AlanineDipeptideImplicit()
implicit_sim = app.Simulation(testsystem.topology,
                              testsystem.system,
                              mm.LangevinIntegrator(temperature * unit.kelvin , 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                              platform=mm.Platform.getPlatformByName('CPU')
                              )
implicit_sim.context.setPositions(testsystem.positions)

openmm_energy = omi.OpenMMEnergyInterface.apply

# Load the training data 
# aldp_traj = mdtraj.load('saved_data/aldp_training_data.h5')
aldp_traj = mdtraj.load('implicit_aldp_training_data.h5')

z = [
    (1, [4, 5, 6]),
    (0, [1, 4, 5]),
    (2, [1, 0, 4]),
    (3, [1, 0, 2]),
    (7, [6, 4, 5]),
    (9, [8, 6, 5]),
    (10, [8, 6, 9]),
    (11, [10, 8, 5]),
    (12, [10, 8, 11]),
    (13, [10, 11, 12]),
    (17, [16, 14, 15]),
    (19, [18, 16, 17]),
    (20, [18, 19, 16]),
    (21, [18, 19, 20])
]
backbone_indices = [4, 5, 6, 8, 14, 15, 16, 18]
aldp_traj.center_coordinates()
ind = aldp_traj.top.select("backbone")
aldp_traj.superpose(aldp_traj, 0, atom_indices=ind, ref_atom_indices=ind)

training_data = aldp_traj.xyz
n_atoms = training_data.shape[1]
n_dim = n_atoms * 3
training_data_npy = training_data.reshape(-1, n_dim)
training_data = torch.from_numpy(training_data_npy)
training_data = training_data.double()

mixed_transform = mixed.MixedTransform(66, backbone_indices, z, training_data)
x, _ = mixed_transform.forward(training_data)
print("Training data shape", x.shape)

dims = 60

class HMC(nn.Module):
    def __init__(self, length):
        super().__init__()
        eps = torch.nn.Parameter(0.01 * torch.ones((length, dims), requires_grad=True))
        log_mass = torch.nn.Parameter(0.0 * torch.ones((length, dims), requires_grad=True))
        self.register_parameter('eps', eps)
        self.register_parameter('log_mass', log_mass)
        self.length = length

    def logP(self, x):
        z, invlogdet = mixed_transform.inverse(x)
        invlogdet = invlogdet.view((invlogdet.shape[0], 1))
        raw_energy = openmm_energy(z, implicit_sim.context, temperature)
        reg_energy = omi.regularize_energy(raw_energy,
            torch.tensor(1e3), torch.tensor(1e8))
        return -reg_energy + invlogdet

    def gradlogP(self, x):
        xp = torch.tensor(x, requires_grad=True)
        assert xp.grad_fn is None
        z = self.logP(xp)
        z.backward(torch.ones((xp.shape[0],1)))
        return xp.grad

    def leapfrog(self, x, p, eps, log_mass):
        # x is pos
        # p is momentum
        for i in range(10):
            p_half = p - (eps / 2.0) * -self.gradlogP(x)
            x = x + eps * (p_half/torch.exp(log_mass))
            p = p_half - (eps / 2.0) * -self.gradlogP(x)
        return x, p

    def draw_samples(self, initial_samples, L=None, return_probs=False):
        x = initial_samples

        if L is None:
            L = self.length

        all_probs = np.zeros((L, initial_samples.shape[0]))
        for i in range(L):
            # Draw momentum
            p = torch.randn_like(x) * torch.exp(0.5 * self.log_mass[i, :])
            # Propose new states
            x_new, p_new = self.leapfrog(x, p, self.eps[i, :], self.log_mass[i, :])

            # Apply M_H
            probs = torch.exp(self.logP(x_new)[:,0] - self.logP(x)[:,0] - \
                0.5*torch.sum(p_new**2 / torch.exp(self.log_mass[i, :]), 1) + \
                0.5*torch.sum(p**2 / torch.exp(self.log_mass[i, :]), 1))
            uniforms = torch.rand_like(probs)
            mask = (uniforms < probs).int()
            mask = torch.transpose(torch.stack(tuple([mask for i in range(60)])), 0, 1)
            x = x_new * mask + x * (1-mask)

            all_probs[i, :] = probs.detach().numpy()

        if return_probs:
            return x, all_probs
        else:
            return x

    def get_log_target_graph(self, initial_samples):
        x = initial_samples
        log_targets = []
        for i in range(self.length):
            # Draw momentum
            p = torch.randn_like(x) * torch.exp(0.5 * self.log_mass[i, :])
            # Propose new states
            x_new, p_new = self.leapfrog(x, p, self.eps[i, :], self.log_mass[i, :])

            # Apply M_H
            probs = torch.exp(self.logP(x_new)[:,0] - self.logP(x)[:,0] - \
                0.5*torch.sum(p_new**2 / torch.exp(self.log_mass[i, :]), 1) + \
                0.5*torch.sum(p**2 / torch.exp(self.log_mass[i, :]), 1))
            uniforms = torch.rand_like(probs)
            mask = (uniforms < probs).int()
            mask = torch.transpose(torch.stack(tuple([mask for i in range(60)])), 0, 1)
            x = x_new * mask + x * (1-mask)
            log_target = -torch.mean(self.logP(x))
            log_targets.append(log_target.detach().numpy())
            if i % 10 == 0:
                print(i)
        return np.array(log_targets)


    def forward(self, initial_samples, L=None):
        if L is None:
            L = self.length
        samples = self.draw_samples(initial_samples, L)
        return -torch.mean(self.logP(samples))



# Define flows
K = 8
#torch.manual_seed(0)

latent_size = 60
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(K):
    s = nf.nets.MLP([latent_size, 4 * latent_size,
        4 * latent_size, latent_size], output_fn='tanh', output_scale=3.0)
    t = nf.nets.MLP([latent_size, 4 * latent_size,
        4 * latent_size, latent_size])
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, s, t)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, s, t)]
    flows += [nf.flows.ActNorm(latent_size)]
flows += [CoordinateTransform(training_data, 66, z, backbone_indices)]

# Set prior and q0
prior = Boltzmann(implicit_sim.context, temperature, energy_cut=1e3,
    energy_max=1e20)
q0 = nf.distributions.DiagGaussian(latent_size)

# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=prior)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)
nfm = nfm.double()

nfm.load_state_dict(torch.load('models/flow_iter_50000'))

def fix_dih(x):
    x = np.where(x<-np.pi, x+2*np.pi, x)
    x = np.where(x>np.pi, x-2*np.pi, x)
    return x


#%%
# ---------- Train different lengths of chain -----------
for length in range(80, 110, 10):
    print(length)
    hmc = HMC(length)
    optimizer = torch.optim.Adam(hmc.parameters(), lr=1e-3)
    training_progress = []
    for i in range(100):
        optimizer.zero_grad()
        flow_samples, _ = nfm.sample(100)
        flow_samples, _ = flows[-1].inverse(flow_samples)
        log_target = hmc.forward(flow_samples.double())
        log_target.backward()
        training_progress.append(log_target.detach().numpy())
        optimizer.step()
    training_progress = np.array(training_progress)
    np.savetxt('saved_data/train_prog_length{}'.format(length), training_progress)
    torch.save(hmc.state_dict(), 'models/hmc_100iter_length{}'.format(length))


#%%
# ----------- Train Flow ---------------
batch_size = 128
ml_losses = np.array([])
optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-3)
for iter in tqdm(range(21000, 100000)):
    optimizer.zero_grad()
    ind = torch.randint(training_data.shape[0], (batch_size,))
    mini_batch = training_data[ind, :].double()
    ml_loss = nfm.forward_kld(mini_batch)
    ml_loss.backward()
    optimizer.step()
    

    ml_losses = np.append(ml_losses, ml_loss.detach().numpy())
    if iter % 1000 == 0:
        plt.plot(ml_losses)
        plt.ylim([-340, -290])
        plt.grid(True)
        plt.show()

    if iter % 5000 == 0:
        torch.save(nfm.state_dict(), 'models/flow_iter_{}'.format(iter))

#%%
# ---------- Test flow marginals ---------
%matplotlib inline
flow_samples, _ = nfm.sample(10000)
flow_samples, _ = flows[-1].inverse(flow_samples)
flow_samples = flow_samples.detach().numpy()
x_np = x.detach().numpy()
import scipy.stats
plot_dims = [23, 32, 35, 41, 53]
for i in plot_dims:
    kde_flow = scipy.stats.gaussian_kde(fix_dih(flow_samples[:, i]))
    kde_data = scipy.stats.gaussian_kde(x_np[:, i])
    positions = np.linspace(-4, 4, 1000)
    plt.plot(positions, kde_data(positions))
    plt.plot(positions, kde_flow(positions))
    plt.show()

#%%
# ---------- KSD ------------
import glob
import scipy.stats
from scipy.spatial.distance import pdist, squareform
# Create dummy HMC just to use it's grad log p function
hmc_dummy = HMC(42)

# HMC
all_samples = np.zeros((7, 100000, 60))
for i, length in enumerate(range(10, 80, 10)):
    all_samples[i, :, :] = np.load(
        'saved_data/hmc_samples/manual_hmc_samples_100000_length_{}.npy'.format(length))

# first estimate the median
medians = np.array([])
for i in range(7):
    medians = np.append(medians, get_median_estimate(all_samples[i, :, :]))
print("medians", medians)
median = np.median(medians)
h_square = 0.5 * median / np.log(100001)
print("h_sqare", h_square)

# get KSDs
ksds = []
for i in range(7):
    samples = torch.tensor(all_samples[i,:,:])
    gradlogp = hmc_dummy.gradlogP(samples)
    ksd = blockKSD(all_samples[i,:,:], gradlogp.detach().numpy(), 10, h_square)
    print("KSD", ksd)
    ksds.append(ksd)

#%%
%matplotlib qt
plt.plot(np.arange(10, 80, 10), ksds)
plt.xlabel('length')
plt.ylabel('ksd')
plt.show()
#%%
ksds = np.array(ksds)
np.savetxt('saved_data/100K_length_ksds', ksds)


# %%
# ---------- Test HMC marginals ------------

# draw samples
# for length in range(10, 80, 10):
# print(length)
# hmc = HMC(length)
# hmc.load_state_dict(torch.load('models/hmc_100iter_length{}'.format(length)))
# flow_samples, _ = nfm.sample(1000)
# flow_samples, _ = flows[-1].inverse(flow_samples)
# hmc_samples = hmc.draw_samples(flow_samples)
# hmc_samples = hmc_samples.detach().numpy()
# np.save('saved_data/length_{}_1000samples'.format(length), hmc_samples)


# flow_samples = flow_samples.detach().numpy()

# %matplotlib inline
%matplotlib qt

x_np = x.detach().numpy()
all_manual_samples = np.zeros((10, 100000, 60))
for i, length in enumerate(range(10, 110, 10)):
    all_manual_samples[i, :, :] = np.load(
        'saved_data/hmc_samples/manual_hmc_samples_100000_length_{}.npy'.format(length))
all_ei_samples = np.zeros((10, 100000, 60))
for i, length in enumerate(range(10, 110, 10)):
    all_ei_samples[i, :, :] = np.load(
        'saved_data/hmc_samples/hmc_samples_100000_length_{}.npy'.format(length))
import scipy.stats
plot_dims = [23, 32, 35, 41, 53]
eps = 1e-5
kls = np.zeros((10, 5))
fig, ax = plt.subplots(4, 5)
kde_eis = []
kde_mans = []
kde_datas = []
for l_id, length in enumerate(range(10, 50, 10)):
    print("**************************")
    print(length)
    # if np.any(np.isnan(all_samples[l_id, :, :])):
    #     print("has nans")
    #     print(np.count_nonzero(np.isnan(all_samples[l_id, :, :])))
    #     all_samples[l_id, np.isnan(all_samples[l_id, :, :])] = 0
    eis = []
    mans = []
    datas = []
    for idx, i in enumerate(plot_dims):
        kde_ei_hmc = scipy.stats.gaussian_kde(fix_dih(all_ei_samples[l_id, :, i]))
        eis.append(kde_ei_hmc)
        kde_man_hmc = scipy.stats.gaussian_kde(fix_dih(all_manual_samples[l_id, :, i]))
        mans.append(kde_man_hmc)
        kde_data = scipy.stats.gaussian_kde(x_np[:, i])
        datas.append(kde_data)
        positions = np.linspace(-4, 4, 1000)
        def kl(x):
            q = kde_data
            p = kde_ei_hmc
            return q(x) * (np.log(q(x) + eps) - np.log(p(x) + eps))
        # kls[l_id, idx] = scipy.integrate.quad(kl, -4, 4)[0]
        ax[l_id, idx].plot(positions, kde_data(positions), label='data')
        ax[l_id, idx].plot(positions, kde_ei_hmc(positions), label='EI')
        ax[l_id, idx].plot(positions, kde_man_hmc(positions), label='manual')
    kde_eis.append(eis)
    kde_mans.append(mans)
    kde_datas.append(datas)
    # print("mean kl", np.mean(kls[l_id, :]))
plt.legend()
plt.show()


#%%
import matplotlib.pyplot as plt
%matplotlib qt
fig, ax = plt.subplots(4, 5)
for i in range(4):
    for j in range(5):
        positions = np.linspace(-4, 4, 1000)
        ax[i, j].plot(positions, kde_datas[i][j](positions), label='data')
        ax[i, j].plot(positions, kde_eis[i][j](positions), label='EI')
        ax[i, j].plot(positions, kde_datas[i][j](positions), label='manual')
plt.legend()
plt.show()

# %%
kls_ei = np.loadtxt('saved_data/100K_KLS_EI.txt')
kls_manual = np.loadtxt('saved_data/100K_KLS_manual.txt')
plt.plot(np.arange(10, 110, 10), np.mean(kls_ei, axis=1), label='ei')
plt.plot(np.arange(10, 110, 10), np.mean(kls_manual, axis=1), label='manual')
plt.legend()
plt.show()



# %%
# ------------ Test manual HMC ---------------
import glob
import scipy.stats
x_np = x.detach().numpy()
plot_dims = [23, 32, 35, 41, 53]
eps = 1e-5
files = glob.glob('saved_data/man_hmc/hmc_samples_1000_length_30_*')
hmc_dummy = HMC(42)

#load
logstuff = np.zeros((100, 2))
all_samples = np.zeros((100, 1000, 60))
for file_idx, file in enumerate(files):
    logeps = float(file[file.find('eps_')+4:file.find('_logmass')])
    logmass = float(file[file.find('logmass_')+8:file.find('.npy')])
    logstuff[file_idx, 0] = logeps
    logstuff[file_idx, 1] = logmass
    all_samples[file_idx, :, :] = np.load(file)


medians = np.array([])
for i in range(100):
    medians = np.append(medians, get_median_estimate(all_samples[i, :, :]))
print("medians", medians)
median = np.median(medians)
h_square = 0.5 * median / np.log(1001)

# get stats
params_vs_stats = np.zeros((100, 4))
for i in range(100):
    print(i)
    samples = torch.tensor(all_samples[i,:,:])
    gradlogp = hmc_dummy.gradlogP(samples)
    ksd = blockKSD(all_samples[i,:,:], gradlogp.detach().numpy(), 10, h_square)

    kls = np.zeros((5,))
    for idx, dim in enumerate(plot_dims):
        kde_hmc = scipy.stats.gaussian_kde(fix_dih(samples[:, dim]))
        kde_data = scipy.stats.gaussian_kde(x_np[:, dim])
        positions = np.linspace(-4, 4, 1000)
        def kl(x):
            q = kde_data
            p = kde_hmc
            return q(x) * (np.log(q(x) + eps) - np.log(p(x) + eps))
        kls[idx] = scipy.integrate.quad(kl, -4, 4)[0]
    params_vs_stats[i, 0] = logstuff[i, 0]
    params_vs_stats[i, 1] = logstuff[i, 1]
    params_vs_stats[i, 2] = np.mean(kls)
    params_vs_stats[i, 3] = ksd
    



#%%

plt.scatter(params_vs_stats[:, 0], params_vs_stats[:, 1],
    c=params_vs_stats[:, 3], s=100)
plt.gray()

plt.show()
id_min = np.argmin(params_vs_stats[:, 2])
print(params_vs_stats[id_min, :])
print(params_vs_stats)





# %%
# ---------- Plot HMC marginals

x_np = x.detach().numpy()
all_manual_samples = np.zeros((10, 100000, 60))
for i, length in enumerate(range(10, 110, 10)):
    all_manual_samples[i, :, :] = np.load(
        'saved_data/hmc_samples/manual_hmc_samples_100000_length_{}.npy'.format(length))
all_ei_samples = np.zeros((10, 100000, 60))
for i, length in enumerate(range(10, 110, 10)):
    all_ei_samples[i, :, :] = np.load(
        'saved_data/hmc_samples/hmc_samples_100000_length_{}.npy'.format(length))
import scipy.stats
l_id = 0
for i in range(60):
    kde_ei_hmc = scipy.stats.gaussian_kde(fix_dih(all_ei_samples[l_id, :, i]))
    kde_man_hmc = scipy.stats.gaussian_kde(fix_dih(all_manual_samples[l_id, :, i]))
    kde_data = scipy.stats.gaussian_kde(x_np[:, i])
    positions = np.linspace(-4, 4, 1000)
    plt.plot(positions, kde_data(positions), label='data')
    plt.plot(positions, kde_ei_hmc(positions), label='EI')
    plt.plot(positions, kde_man_hmc(positions), label='manual')
    plt.legend()
    plt.show()


# %%
# ------------ Plots KLS against each other -----------
%matplotlib qt
ei_data = np.loadtxt('saved_data/100K_KLS_EI.txt')
man_data = np.loadtxt('saved_data/100K_KLS_manual.txt')

plt.plot(np.arange(10, 110, 10), np.mean(ei_data, 1), label='EI')
plt.plot(np.arange(10, 110, 10), np.mean(man_data, 1), label='manual')
plt.legend()
plt.xlabel('HMC length')
plt.ylabel('Average KL divergence')
plt.show()
#%%
# -------- Plots KSDS against each other ----------
%matplotlib qt
ei_data = np.loadtxt('saved_data/100K_length_ksds_ei')
man_data = np.loadtxt('saved_data/100K_length_ksds_manual')
plt.plot(np.arange(10, 80, 10), ei_data, label='EI')
# plt.plot(np.arange(10, 80, 10), man_data, label='manual')
plt.plot(np.arange(10, 80, 10), 0.005239*np.ones(7), label='Training data',
    linestyle = '--')
plt.plot(np.arange(10, 80, 10), 0.02297*np.ones(7),
    label='Manually tuned at length 30', linestyle='--')
plt.legend(loc= 'center right')
plt.xlabel('HMC length')
plt.ylabel('KSD')
plt.show()



# %%
# ---------- See how the training does on KSD ----------
dummy_hmc = HMC(42)

x_np = x.detach().numpy()
median = get_median_estimate(x_np)
h_square = 0.5 * median / np.log(100001)
print("h_square", h_square)

gradlogp = dummy_hmc.gradlogP(x)
ksd = blockKSD(x_np, gradlogp.detach().numpy(), 10, 5.6015)

print(ksd)


# %%
# ---------- Subsamples the manual HMCs KSD -----------
import glob
import scipy.stats
x_np = x.detach().numpy()
files = glob.glob('saved_data/grid_man_samples/*')
print(len(files))
hmc_dummy = HMC(42)

#load
logstuff = np.zeros((47, 2))
all_samples = np.zeros((47, 3000, 60))
for file_idx, file in enumerate(files):
    logeps = float(file[file.find('eps_')+4:file.find('_logmass')])
    logmass = float(file[file.find('logmass_')+8:file.find('.npy')])
    logstuff[file_idx, 0] = logeps
    logstuff[file_idx, 1] = logmass
    all_samples[file_idx, :, :] = np.load(file)

h_square = 5.6015

# get stats
params_vs_stats = np.zeros((47, 5))
for i in range(47):
    print(i)
    samples = torch.tensor(all_samples[i,:,:])
    gradlogp = hmc_dummy.gradlogP(samples).detach().numpy()
    params_vs_stats[i, 2] = KSD(all_samples[i,0:1000,:],
        gradlogp[0:1000,:], h_square)
    params_vs_stats[i, 3] = KSD(all_samples[i,1000:2000,:],
        gradlogp[1000:2000,:], h_square)
    params_vs_stats[i, 4] = KSD(all_samples[i,2000:3000,:],
        gradlogp[2000:3000,:], h_square)
    params_vs_stats[i, 0] = logstuff[i, 0]
    params_vs_stats[i, 1] = logstuff[i, 1]

np.savetxt('saved_data/man_hmc_length_30_grid_KSDS_3repeats_of_1000_v3.txt',
    params_vs_stats)

# %%
# ---------- Check 100K manual tune KSD ----------
dummy_hmc = HMC(42)

samples_np = np.load('saved_data/ksd_man_hmc/100K_-6logeps_-1_5logmass.npy')
samples = torch.tensor(samples_np)
gradlogp = dummy_hmc.gradlogP(samples)
ksd = blockKSD(samples_np, gradlogp.detach().numpy(), 10, 5.6015)

print(ksd)


# %%
# ----------- break 100K manual tune KSD into 10 and compare to EI ------
dummy_hmc = HMC(42)
man_samples_np = np.load('saved_data/ksd_man_hmc/100K_-6logeps_-1_5logmass.npy')
ei_samples_np = np.load('saved_data/hmc_samples/hmc_samples_100000_length_30.npy')
man_samples = torch.tensor(man_samples_np)
ei_samples = torch.tensor(ei_samples_np)
x_np = x.detach().numpy()
flow_samples, _ = nfm.sample(100000)
flow_samples, _ = flows[-1].inverse(flow_samples)
flow_samples_np = flow_samples.detach().numpy()

ei_ksds = []
man_ksds = []
x_ksds = []
flow_ksds = []
for i in range(10):
    ei_gradlogp = dummy_hmc.gradlogP(ei_samples[i*10000:(i+1)*10000,:]).detach().numpy()
    man_gradlogp = dummy_hmc.gradlogP(man_samples[i*10000:(i+1)*10000,:]).detach().numpy()
    x_gradlogp = dummy_hmc.gradlogP(x[i*10000:(i+1)*10000,:]).detach().numpy()
    flow_gradlogp = dummy_hmc.gradlogP(flow_samples[i*10000:(i+1)*10000,:]).detach().numpy()
    ei_ksd = KSD(ei_samples_np[i*10000:(i+1)*10000,:], ei_gradlogp, 5.6015)
    man_ksd = KSD(man_samples_np[i*10000:(i+1)*10000,:], man_gradlogp, 5.6015)
    x_ksd = KSD(x_np[i*10000:(i+1)*10000,:], x_gradlogp, 5.6015)
    flow_ksd = KSD(flow_samples_np[i*10000:(i+1)*10000,:], flow_gradlogp, 5.6015)
    print("EI", ei_ksd)
    print("man", man_ksd)
    print("x", x_ksd)
    print("flow", flow_ksd)
    ei_ksds.append(ei_ksd)
    man_ksds.append(man_ksd)
    x_ksds.append(x_ksd)
    flow_ksds.append(flow_ksd)
ei_ksds=np.array(ei_ksds)
man_ksds=np.array(man_ksds)
x_ksds = np.array(x_ksds)
flow_ksds = np.array(flow_ksds)
# %%
from matplotlib import rc
%matplotlib qt
plt.figure(figsize=(5,5))
# rc('text', usetex=True)
# rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rcParams.update({'font.size': 20})
# data = np.zeros((4, 10))
data = np.load('tempdata.npy')
# data[0,:]=man_ksds
# data[1,:]=ei_ksds
# data[2,:]=x_ksds
# data[3, :] = flow_ksds
# np.save('tempdata', data)

lqs = np.percentile(data, 25, axis=1)
print(lqs)
uqs = np.percentile(data, 75, axis=1)
print(uqs)
medians = np.median(data, axis=1)
print(medians)
errors = np.zeros((2, 4))
errors[0, :] = medians - lqs
errors[1, :] = uqs - medians
plt.errorbar([0,1], medians[[0, 1]], errors[:, [0, 1]], fmt='kx', capsize=10, linewidth=3,
    elinewidth=2, markersize=10, label='Median')
# plt.figure(figsize=(3,5))
# plt.plot(data,'x',color='black')
# plt.xticks([0, 1,2,3], ['Manual', 'EI', 'Training data', 'flow'])
plt.xticks([0,1], ['Manual', 'EI'])
plt.ylabel('KSD')
plt.tight_layout()
plt.show()

# %%
print(np.median(data, axis=1))


# %%

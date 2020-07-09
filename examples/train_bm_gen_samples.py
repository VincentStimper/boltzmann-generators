# Import modules
import torch
import numpy as np

import boltzgen as bg

import argparse
import os
from time import time


# Parse input arguments
parser = argparse.ArgumentParser(description='Train Stochastic Normalizing Flow as a Boltzmann Generator')

parser.add_argument('--config', type=str, help='Path config file specifying model architecture and training procedure',
                    default='../config/bm.yaml')
parser.add_argument("--resume", action="store_true", help='Flag whether to resume training')
parser.add_argument("--tlimit", type=float, default=None,
                    help='Number of hours after which to stop training')

args = parser.parse_args()


# Load config
config = bg.utils.get_config(args.config)


# Create model
model = bg.BoltzmannGenerator(config)

# Move model on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.double()


# Load training data
training_data = bg.utils.load_traj(config['data_path'])

# Initialize ActNorm layers
batch_size = config['train']['batch_size']
ind = torch.randint(len(training_data), (batch_size,))
x = training_data[ind, :].double().to(device)
kld = model.forward_kld(x)



# Train model
max_iter = config['train']['max_iter']
n_data = len(training_data)
checkpoint_step = config['train']['checkpoint_iter']
checkpoint_root = config['train']['checkpoint_root']

loss_log = None

optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'],
                             weight_decay=config['train']['weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=config['train']['rate_decay'])

# Initialize model if desired
if config['train']['init_model'] is not None:
    model.load(config['train']['init_model'])

# Resume training if needed
start_iter = 0
if args.resume:
    latest_cp = bg.utils.get_latest_checkpoint(os.path.join(checkpoint_root, 'checkpoints'),
                                               'model')
    if latest_cp is not None:
        model.load(latest_cp)
        optimizer_path = os.path.join(checkpoint_root, 'checkpoints/optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        loss_path = os.path.join(checkpoint_root, 'log/loss.csv')
        if os.path.exists(loss_path):
            loss_log = np.loadtxt(loss_path, delimiter=',', skiprows=1)
        start_iter = int(latest_cp[-8:-3])
if start_iter > 0:
    for _ in range(start_iter // config['train']['decay_iter']):
        lr_scheduler.step()

start_time = time()

for it in range(start_iter, max_iter):
    optimizer.zero_grad()
    fkld = model.forward_kld(x)
    z, logq = model.sample(batch_size)
    logp = model.p.log_prob(z)

    # Assemble loss
    loss = 0
    loss_log_ = [logq.to('cpu').data.numpy(), logp.to('cpu').data.numpy()]
    header_log = 'loss,logq,logp'
    if config['train']['fkld']['coeff'] > 0 or config['train']['fkld']['log']:
        ind = torch.randint(n_data, (batch_size,))
        x = training_data[ind, :].double().to(device)
        fkld = model.forward_kld(x)
        if config['train']['fkld']['log']:
            loss_log_.append(fkld.to('cpu').data.numpy())
            header_log += ',fkld'
        if config['train']['fkld']['coeff'] > 0:
            loss = loss + config['train']['fkld']['coeff'] * fkld
    if config['train']['rkld']['coeff'] > 0 or config['train']['rkld']['log']:
        if 'annealing' in config['train']['rkld'] \
                and config['train']['rkld']['annealing']['type'] is not None:
            anneal_len = config['train']['rkld']['annealing']['length']
            if config['train']['rkld']['annealing']['type'] == 'lin':
                beta = np.linspace(config['train']['rkld']['annealing']['start'], 1,
                                   anneal_len)
                beta = np.concatenate([beta, np.ones(max_iter - anneal_len)])
            elif config['train']['rkld']['annealing']['type'] == 'geom':
                beta = np.geomspace(config['train']['rkld']['annealing']['start'], 1,
                                    anneal_len)
                beta = np.concatenate([beta, np.ones(max_iter - anneal_len)])
            else:
                raise NotImplementedError('The annealing type '
                                          + config['train']['rkld']['annealing']['type']
                                          + ' is not yet implemented')
        else:
            beta = np.ones(max_iter)
        rkld = torch.mean(logq) - beta[it] * torch.mean(logp)
        if config['train']['rkld']['log']:
            loss_log_.append(rkld.to('cpu').data.numpy())
            header_log += ',rkld'
        if config['train']['rkld']['coeff'] > 0:
            loss = loss + config['train']['rkld']['coeff'] * rkld
    if config['train']['alphadiv']['coeff'] > 0 or config['train']['alphadiv']['log']:
        alphadiv = -torch.logsumexp(config['train']['alphadiv']['alpha'] * (logp - logq), 0) \
                   + np.log(logp.shape[0])
        if config['train']['alphadiv']['log']:
            header_log += ',alphadiv'
            loss_log_.append(alphadiv.to('cpu').data.numpy())
        if config['train']['alphadiv']['coeff'] > 0:
            loss = loss + config['train']['alphadiv']['coeff'] * alphadiv
    if config['train']['angle_loss']['coeff'] > 0 or config['train']['angle_loss']['log']:
        angle_loss = torch.mean(model.flows[-1].mixed_transform.ic_transform.angle_loss)
        if config['train']['angle_loss']['log']:
            header_log += ',angle_loss'
            loss_log_.append(angle_loss.to('cpu').data.numpy())
        if config['train']['angle_loss']['coeff'] > 0:
            loss = loss + config['train']['angle_loss']['coeff'] * angle_loss

    loss_log_ = [loss.to('cpu').data.numpy()] + loss_log_
    if loss_log is None:
        loss_log = np.array(loss_log_)[None, :]
    else:
        loss_log = np.concatenate([loss_log, loss_log_], 0)

    if not torch.isnan(loss) and loss < 0:
        loss.backward()
        optimizer.step()
    
    if (it + 1) % checkpoint_step == 0:
        model.save(os.path.join(checkpoint_root, 'checkpoints/model_%05i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(checkpoint_root, 'checkpoints/optimizer.pt'))
        np.savetxt(os.path.join(checkpoint_root, 'log/loss.csv'), loss_log, delimiter=',',
                   header=header_log, comments='')
        if args.tlimit is not None and (time() - start_time) / 3600 > args.tlimit:
            break
    
    if (it + 1) % config['train']['decay_iter'] == 0:
        lr_scheduler.step()
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

loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'],
                             weight_decay=config['train']['weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=config['train']['rate_decay'])

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
            loss_hist = np.loadtxt(loss_path)
        start_iter = int(latest_cp[-10:-3])
if start_iter > 0:
    for _ in range(start_iter // config['train']['decay_iter']):
        lr_scheduler.step()

if 'loss_limit' in config['train']:
    loss_limit = config['train']['loss_limit']
else:
    loss_limit = 1e100

start_time = time()

for it in range(start_iter, max_iter):
    optimizer.zero_grad()
    ind = torch.randint(n_data, (batch_size, ))
    x = training_data[ind, :].double().to(device)
    fkld = model.forward_kld(x)
    if 'angle_loss' in config['train'] and config['train']['angle_loss']:
        loss = fkld + torch.mean(model.flows[-1].mixed_transform.ic_transform.angle_loss)
    else:
        loss = fkld
    if not torch.isnan(loss) and loss < loss_limit:
        loss.backward()
        optimizer.step()
    
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    if (it + 1) % checkpoint_step == 0:
        model.save(os.path.join(checkpoint_root, 'checkpoints/model_%07i.pt' % (it + 1)))
        torch.save(optimizer.state_dict(),
                   os.path.join(checkpoint_root, 'checkpoints/optimizer.pt'))
        np.savetxt(os.path.join(checkpoint_root, 'log/loss.csv'), loss_hist)
        if args.tlimit is not None and (time() - start_time) / 3600 > args.tlimit:
            break
    
    if (it + 1) % config['train']['decay_iter'] == 0:
        lr_scheduler.step()
    
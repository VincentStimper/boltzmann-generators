# Import modules
import torch
import numpy as np

import boltzgen as bg

import argparse


# Parse input arguments
parser = argparse.ArgumentParser(description='Train Stochastic Normalizing Flow as a Boltzmann Generator')

parser.add_argument('--config', type=str, help='Path config file specifying model architecture and training procedure',
                    default='../config/bm.yaml')

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
checkpoint_path = config['train']['checkpoint_root']

loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'],
                             weight_decay=config['train']['weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['train']['rate_decay'])
for it in range(max_iter):
    optimizer.zero_grad()
    ind = torch.randint(n_data, (batch_size, ))
    x = training_data[ind, :].double().to(device)
    fkld = model.forward_kld(x)
    loss = fkld
    if not torch.isnan(loss) and loss < 0:
        loss.backward()
        optimizer.step()
    
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    if (it + 1) % checkpoint_step == 0:
        model.save(checkpoint_path + 'checkpoints/model_%05i.pt' % (it + 1))
        torch.save(optimizer.state_dict(), checkpoint_path + 'checkpoints/optimizer.pt')
        np.savetxt(checkpoint_path + 'log/loss.csv', loss_hist)
    
    if (it + 1) % config['train']['decay_iter'] == 0:
        lr_scheduler.step()
    
import os
import shutil

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import tqdm
import yaml

from dataset import ODIRDataset
from SDAE import TwinSDAE


def batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch


def run_sdae(configs, dataset, writer, name='sdae', device=None):
    os.makedirs('./checkpoints/{}/', exist_ok=True)

    # split into training and testing
    test_len = int(configs['test_fraction'] * len(dataset))
    train_len = len(dataset) - test_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    # get dataloaders
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], pin_memory=True)

    # get model, optimizer and loss function
    def loss_fn(out):
        loss = 0.0
        for key in out:
            embs, recons = out[key]
            for i in range(len(recons)):
                loss += torch.mean((embs[i] - recons[i]) ** 2)
        return loss

    model = TwinSDAE(3, emb_channels=configs['emb_channels'], noise_std=configs['noise_std'],
        share_weights=configs['share_weights'])
    model = model.to(device)
    optimizer = Adam(model.parameters(), configs['lr'])
    
    iters = 0
    for epoch in range(configs['n_epochs']):
        train_losses, test_losses = [], []

        # training loop
        for batch in tqdm.tqdm(train_loader):
            batch = batch_to_device(batch, device)
            out = model(batch['left_image'], batch['right_image'])

            train_loss = loss_fn(out)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.detach().item())
            writer.add_scalars('loss', {'train': train_losses[-1]}, iters)
            iters += 1

            # evaluate on test set
            if iters % configs['test_freq'] == 0:
                model.eval()
                test_losses = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch_to_device(batch, device)
                        out = model(batch['left_image'], batch['right_image'])
                        test_loss = loss_fn(out).item()
                        test_losses.append(test_loss)
                    writer.add_scalars('loss', {'test': sum(test_losses) / len(test_losses)}, iters)
                model.train()

                # print results
                print('epoch {}\n'
                      '  train loss = {}\n'
                      '  test loss  = {}'.format(
                        epoch, sum(train_losses) / len(train_losses), sum(test_losses) / len(test_losses))) 

        # save checkpoint
        checkpoint_path = os.path.join('./checkpoints/{}/{}.ckpt'.format(
            name, epoch))
        torch.save(model, checkpoint_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='',
        help='Experiment name.')
    parser.add_argument('--configuration-file', '-cf', type=str, required=True,
        help='Path to configuration file with hyperparameters.')
    parser.add_argument('--csv-file', '-csv', type=str, default='./data/full_df.csv',
        help='CSV file containing ODIR-5K dataset information.')
    parser.add_argument('--image-dir', '-id', type=str, default='./data/preprocessed_images/',
        help='Root of the directory containing the dataset images.')
    args = parser.parse_args()

    # make output directories if not present
    os.makedirs('./logs/', exist_ok=True)
    os.makedirs('./checkpoints/', exist_ok=True)

    # open experiment configuration file
    with open(args.configuration_file, 'r') as cf:
        configs = yaml.load(cf)

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available

    dataset = ODIRDataset(args.csv_file, args.image_dir, configs['image_size']) # load dataset

    name = args.name
    if not name:
        name = configs['model']
    
    shutil.rmtree('./logs/{}/'.format(name)) # clear previous logs with the same experiment name
    writer = SummaryWriter(log_dir='./logs/{}/'.format(name))

    # run experiment
    if configs['model'] == 'sdae':
        run_sdae(configs, dataset, writer, name=name, device=device)
    
    writer.close()
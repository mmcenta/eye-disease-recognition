from collections import defaultdict
import pickle
import os
import shutil

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import tqdm
import yaml

from classification import SDAEClassification, ResNet50Classification
from dataset import ODIRDataset
from SDAE import TwinSDAE
from utils import batch_to_device, update_metrics, write_logs, print_last_logs, get_cv_indices


def run_sdae_pretraining(configs, dataset, writer, name, device):
    os.makedirs('./checkpoints/{}/'.format(name), exist_ok=True)

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
    
    iters, train_losses = 0, []
    for epoch in range(configs['n_epochs']):
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
                train_losses = []

        # save checkpoint
        checkpoint_path = os.path.join('./checkpoints/{}/{}.ckpt'.format(
            name, epoch))
        torch.save(model, checkpoint_path)


def run_linear_classification(configs, dataset, train_loader, test_loader,
    model, writer, name, device):

    print('Starting inear finetuning with name {}'.format(name))
    os.makedirs('./checkpoints/{}/'.format(name), exist_ok=True)

    # prepare model, loss, and optimizer
    model = model.to(device)
    if configs['target_col'] < 0:
        loss_fn = nn.CrossEntropyLoss(weight=dataset.get_weight().to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=dataset.get_weight())
    optimizer = Adam(model.parameters(), configs['lr'])

    iters, metrics, logs = 0, defaultdict(list), defaultdict(list)
    for epoch in range(configs['n_epochs']):
        # training loop
        for batch in tqdm.tqdm(train_loader):
            batch = batch_to_device(batch, device)
            logits = model(batch['left_image'], batch['right_image'])

            loss = loss_fn(logits, batch['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters += 1

            update_metrics(metrics, loss, logits, batch['target'], training=True)

            # evaluate on test set
            if iters % configs['test_freq'] == 0:
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch_to_device(batch, device)
                        logits = model(batch['left_image'], batch['right_image'])
                        loss = loss_fn(logits, batch['target'])
                        update_metrics(metrics, loss, logits, batch['target'], training=False)

                write_logs(logs, iters, metrics, writer)
                model.train()

                # print results
                print_last_logs(epoch, logs)

        # save checkpoint
        checkpoint_path = os.path.join('./checkpoints/{}/{}.ckpt'.format(
            name, epoch))
        torch.save(model, checkpoint_path)

    return logs


def run_cv_finetuning(configs, model_fn, dataset, writer, name, device):
    for cv, (train_indices, test_indices) in enumerate(get_cv_indices(dataset, configs['n_folds'])):
        # load model
        model = model_fn()

        # prepare data loaders
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(dataset, batch_size=configs['batch_size'], 
            sampler=train_sampler, pin_memory=True)
        test_loader = DataLoader(dataset, batch_size=configs['batch_size'], 
            sampler=test_sampler, pin_memory=True)

        # run experiment
        logs = run_linear_classification(configs, dataset, train_loader, test_loader,
                    model, writer, '{}_cv{}'.format(name, cv), device)
        
        with open('./logs/{}/cv_{}_logs.pkl'.format(name, cv), 'wb') as f:
            pickle.dump(logs, f)


def run_sdae_finetuning(configs, dataset, writer, name, device):
    print('Fine-tuning SDAE model from checkpoint {}...'.format(configs['sdae_checkpoint']))

    # build classification model
    def base_model_fn():
        model = torch.load(configs['sdae_checkpoint'], map_location=device)
        fake_image = torch.randn(1, 3, configs['image_size'], configs['image_size'])
        out = model(fake_image, fake_image)
        emb_size = out['left'][0][-1].size()
        return model, emb_size

    n_classes = 8 if configs['target_col'] < 0 else 1
    model_fn = lambda: SDAEClassification(n_classes, base_model_fn, freeze=configs['freeze'])
    
    run_cv_finetuning(configs, model_fn, dataset, writer, name, device)


def run_resnet50_finetuning(configs, dataset, writer, name, device):
    print('Fine-tuning ResNet-50 model...')

    # build classification model
    def base_model_fn():
        model = torchvision.models.resnet50(pretrained=True)
        emb_size = (model.fc.in_features,)
        return model, emb_size

    n_classes = 8 if configs['target_col'] < 0 else 1
    model_fn = lambda: ResNet50Classification(n_classes, base_model_fn, freeze=configs['freeze'])
    
    run_cv_finetuning(configs, model_fn, dataset, writer, name, device)


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

    dataset = ODIRDataset(args.csv_file, args.image_dir, configs['image_size'],
        configs['target_col'], balanced=configs['balanced'],
        normalize=configs['normalize'])

    name = args.name
    if not name:
        name = configs['model']
    
    log_dir = './logs/{}/'.format(name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir) # clear previous logs with the same experiment name
    writer = SummaryWriter(log_dir='./logs/{}/'.format(name))

    # run experiment
    if configs['model'] == 'sdae-pre':
        run_sdae_pretraining(configs, dataset, writer, name, device)
    elif configs['model'] == 'sdae':
        run_sdae_finetuning(configs, dataset, writer, name, device)
    elif configs['model'] == 'resnet50':
        run_resnet50_finetuning(configs, dataset, writer, name, device)
    
    writer.close()
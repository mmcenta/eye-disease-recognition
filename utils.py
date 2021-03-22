from collections import defaultdict
from functools import partial, reduce

import numpy as np
from numpy.core.numeric import indices
from sklearn import metrics
import torch


def batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch


def accuracy(logits, target):
    pred = torch.squeeze(torch.sigmoid(logits) > 0.5).long()
    return metrics.accuracy_score(target, pred)


def top_k_accuracy(logits, target, k=3):
    n_classes = logits.size(1)
    score = torch.softmax(logits, -1)
    return metrics.top_k_accuracy_score(target, score, k=k,
        labels=np.arange(n_classes))


def roc_auc(logits, target):
    n_classes = logits.size(1)
    if n_classes > 1:
        score = torch.softmax(logits, -1)
    else:
        score = torch.sigmoid(logits)
    return metrics.roc_auc_score(target, score, multi_class='ovr',
        labels=np.arange(n_classes))


BINARY_METRICS_TO_FN = {
    'accuracy': accuracy,
    'roc_auc': roc_auc,
}

MULTI_METRICS_TO_FN = {
    'accuracy': partial(top_k_accuracy, k=1),
    'top_3_accuracy': partial(top_k_accuracy, k=3),
    'roc_auc': roc_auc,
}


def update_metrics(metrics, loss, logits, target, training=True):
    loss, logits = loss.detach(), logits.detach()
    prefix = 'train_' if training else 'test_'
    metrics[prefix + 'loss'].append(loss.detach().item())
    metrics_to_fn = MULTI_METRICS_TO_FN if logits.size(1) > 1 else BINARY_METRICS_TO_FN 
    for key, fn in metrics_to_fn.items():
        try:
            metrics[prefix + key].append(fn(logits, target))
        except ValueError:
            pass


def write_logs(logs, iters, metrics, writer, binary=False):
    metrics_to_fn = BINARY_METRICS_TO_FN if binary else MULTI_METRICS_TO_FN
    for prefix in ('train', 'test'):
        for key in ['loss'] + list(metrics_to_fn.keys()):
            buffer = metrics[prefix + '_' + key]
            if len(buffer) == 0:
                logs[prefix + '_' + key].append(None)

            value = sum(buffer) / len(buffer)

            logs[prefix + '_' + key].append(value)
            writer.add_scalars(key, {prefix: value}, iters)

            metrics[prefix + '_' + key] = []


def print_last_logs(epoch, logs):
    print('epoch {}:'.format(epoch))
    for key, values in logs.items():
        print('  {} = {}'.format(key, values[-1]))


def get_cv_indices(dataset, folds=10):
    """Returns a generator of stratified cross-validation train and test indices."""
    df = dataset._df

    # separate examples of each class
    indices_per_class = defaultdict(list)
    for idx in range(len(df)):
        item = df.iloc[idx]
        y = np.argmax(list(map(int, item['target'][1:-1].split(','))))
        indices_per_class[y].append(idx)
    
    # shuffle each class
    for y in indices_per_class:
        indices_per_class[y] = np.array(indices_per_class[y], dtype=np.long)
        np.random.shuffle(indices_per_class[y])

    # create partitions per class
    partitions_per_class = {}
    for y, indices in indices_per_class.items():
        fold_size = 1 + int(len(indices) / folds)
        partitions_per_class[y] = [indices[k * fold_size: min((k + 1) * fold_size, len(indices))] for k in range(folds)]

    # mix and shuffle partitions
    partitions = []
    for i in range(folds):
        partition = reduce(lambda p, t: np.concatenate((p, t)),
            [partitions_per_class[y][i] for y in partitions_per_class])
        partitions.append(partition)

    # generate stratified cross validation splits
    for i in range(folds):
        test_indices = partitions[i]
        train_indices = reduce(lambda p, t: np.concatenate((p, t)), partitions[:i] + partitions[i + 1:])
        yield train_indices, test_indices


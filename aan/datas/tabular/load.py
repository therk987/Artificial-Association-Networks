"""Iris tabular dataset (paper, Experiment 5, 'tabular').

Replicates the legacy loader exactly: sklearn's bundled Iris data, stratified
150 -> 120/30 test split then 120 -> 96/24 valid split (both
``train_test_split(test_size=0.2, random_state=0, stratify=y)``), features
standardized by a ``StandardScaler`` fit on the 120-sample pre-valid training
portion.
"""
import torch


def IRIS_DATA(test_size=0.2, random_state=0):
    """Return {'train'|'valid'|'test': (FloatTensor (n,4), LongTensor (n,))}."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    x, y = iris['data'], iris['target']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=test_size, random_state=random_state,
        stratify=y_train)

    result = {
        'train': (torch.tensor(x_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
        'valid': (torch.tensor(x_valid, dtype=torch.float32),
                  torch.tensor(y_valid, dtype=torch.long)),
        'test': (torch.tensor(x_test, dtype=torch.float32),
                 torch.tensor(y_test, dtype=torch.long)),
    }
    print('LOAD : IRIS  train {} valid {} test {}'.format(
        len(result['train'][0]), len(result['valid'][0]),
        len(result['test'][0])), flush=True)
    return result

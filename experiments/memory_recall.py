"""M1 — episodic binding: can the shared cell RETRIEVE, not memorize?

Every episode draws FRESH random class prototypes, so class identity exists
only in that episode's memory contents — weights cannot store the answer.
The model must bind the noisy query to the right memory slots through the
recall tree (star adjacency) built by aan.memory.episodic. Memory slots
carry their label as a one-hot appended to the vector; the query carries
zeros there.

    no-memory tree   -> chance (1/K) by construction
    working retrieval -> high accuracy

Compares cell versions (the tau attention aggregation is literal key-value
retrieval; gau/gaau retrieve through their graph aggregations):

    python experiments/memory_recall.py --versions tau,gaau,gau --seeds 0 1 2
"""
import argparse
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from aan.config.option import device as default_device
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.memory.episodic import EpisodicMemory, QUERY_DOMAIN, MEMORY_DOMAIN
from aan.models.feature_encoders.multiencoder_connector import MultiExtractionConnector
from aan.models.feature_encoders.domains.vector2vec import VectorEncoder
from aan.models.encoders.flat_recursive_encoder import FlatRecursiveAssociationNeuralNetworks


def episode_batch(batch_size, k, m, dim, noise_mem, noise_query, with_memory=True):
    """One BatchNeuroTree of recall trees + targets, fresh prototypes each."""
    trees, targets = [], []
    for _ in range(batch_size):
        prototypes = torch.randn(k, dim)
        memory = EpisodicMemory(capacity_per_class=m)
        for label in range(k):
            vecs = prototypes[label] + noise_mem * torch.randn(m, dim)
            tagged = torch.cat([vecs, torch.eye(k)[label].repeat(m, 1)], dim=-1)
            memory.write(tagged, [label] * m)
        target = int(torch.randint(k, (1,)))
        query_vec = prototypes[target] + noise_query * torch.randn(dim)
        query = torch.cat([query_vec, torch.zeros(k)], dim=-1)
        if with_memory:
            tree = memory.recall_tree(query)
        else:
            from aan.data_structures.neuronode import NeuroNode
            tree = NeuroNode(None, None, C=[NeuroNode(query, QUERY_DOMAIN)])
        tree.reset_state()
        trees.append(tree)
        targets.append(target)
    return BatchNeuroTree(trees), torch.tensor(targets)


def build_model(version, vec_dim, feature_dim, hidden_dim, k, device):
    fe = MultiExtractionConnector(feature_dim, {
        QUERY_DOMAIN: VectorEncoder(vec_dim, feature_dim),
        MEMORY_DOMAIN: VectorEncoder(vec_dim, feature_dim),
    })
    engine = FlatRecursiveAssociationNeuralNetworks(
        feature_dim, hidden_dim, fe, None, version=version)
    head = nn.Linear(hidden_dim, k)
    return engine.to(device), head.to(device)


def run(version, seed, args, with_memory=True):
    torch.manual_seed(seed)
    device = args.device
    vec_dim = args.dim + args.classes
    engine, head = build_model(version, vec_dim, args.feature_dim,
                               args.hidden_dim, args.classes, device)
    opt = torch.optim.Adam(list(engine.parameters()) + list(head.parameters()),
                           lr=args.lr)

    for step in range(args.steps):
        batch, targets = episode_batch(args.batch_size, args.classes, args.per_class,
                                       args.dim, args.noise_mem, args.noise_query,
                                       with_memory)
        logits = head(engine(batch))
        loss = F.cross_entropy(logits, targets.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

    engine.eval(), head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            batch, targets = episode_batch(args.batch_size, args.classes,
                                           args.per_class, args.dim,
                                           args.noise_mem, args.noise_query,
                                           with_memory)
            pred = head(engine(batch)).argmax(dim=-1).cpu()
            correct += int((pred == targets).sum())
            total += len(targets)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', default='tau,gaau,gau')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--per-class', type=int, default=2)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--feature-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--noise-mem', type=float, default=0.1)
    parser.add_argument('--noise-query', type=float, default=0.3)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default=default_device)
    args = parser.parse_args()

    print('chance = %.3f (K=%d, fresh prototypes per episode)'
          % (1.0 / args.classes, args.classes))

    baseline = [run('gau', seed, args, with_memory=False) for seed in args.seeds]
    print('%-18s %.4f +/- %.4f' % ('no-memory (gau)', statistics.mean(baseline),
                                   statistics.stdev(baseline) if len(baseline) > 1 else 0.0))

    for version in [v.strip() for v in args.versions.split(',') if v.strip()]:
        accs = [run(version, seed, args) for seed in args.seeds]
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print('%-18s %.4f +/- %.4f' % ('recall (%s)' % version,
                                       statistics.mean(accs), std))


if __name__ == '__main__':
    main()

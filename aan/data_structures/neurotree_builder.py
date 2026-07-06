from aan.data_structures.batch_neurotree import BatchNeuroTree


def neurotreeBuilder(batch_x, domains, subtasks, data2neurotree):
    """Build a BatchNeuroTree from raw samples.

    ``data2neurotree`` maps a domain name to a builder function
    ``(x, domain) -> NeuroNode``.
    """
    neurotrees = [
        data2neurotree[domain](x, domain)
        for x, domain in zip(batch_x, domains)
    ]
    return BatchNeuroTree(neurotrees)

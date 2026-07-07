"""E6 — structural ablation transforms (the narrowed-novelty proof).

The revision's novelty claim is that neurotrees add (1) multiple parents and
(2) level jumps (a parent may consume a hidden state produced any number of
levels below — the data-structured skip connection) on top of plain trees.
E6 degrades real neurotrees along exactly these axes and measures the drop:

    to_single_parent(tree)     — every shared node is deep-copied per parent,
                                 so the DAG becomes a plain tree (no hidden
                                 state reuse across parents)
    insert_level_padding(tree) — every edge that skips levels gets empty
                                 pass-through nodes, so information can only
                                 climb one level at a time (no jumps)

Both preserve x / domain / A_c so the information content is unchanged;
only the propagation structure is degraded.
"""
from aan.data_structures.neuronode import NeuroNode


def _fields(node):
    """Constructor kwargs, tolerant of legacy pickled nodes (missing attrs)."""
    return dict(s_t=getattr(node, 's_t', None), A_c=getattr(node, 'A_c', None),
                E_c=getattr(node, 'E_c', None))


def _copy_meta(clone, node):
    clone.y = getattr(node, 'y', None)
    clone.label = getattr(node, 'label', None)


def _clone_subtree(node, memo=None):
    clone = NeuroNode(node.x, node.t_d, **_fields(node))
    _copy_meta(clone, node)
    for child in node.C:
        clone.insert(_clone_subtree(child))
    return clone


def to_single_parent(root):
    """Deep-copy every re-visited node so each node has exactly one parent.

    Returns a NEW tree; the original is untouched. Node count grows by the
    amount of sharing the neurotree had (that growth is itself reported by
    count_nodes as an E6 statistic).
    """
    seen = set()

    def walk(node):
        if id(node) in seen:
            return _clone_subtree(node)
        seen.add(id(node))
        rebuilt = NeuroNode(node.x, node.t_d, **_fields(node))
        _copy_meta(rebuilt, node)
        for child in node.C:
            rebuilt.insert(walk(child))
        return rebuilt

    return walk(root)


def _height(node, memo):
    nid = id(node)
    if nid in memo:
        return memo[nid]
    h = 1 + max((_height(c, memo) for c in node.C), default=-1)
    memo[nid] = h
    return h


def insert_level_padding(root):
    """Insert empty association nodes on every level-jumping edge.

    After this, every parent-child edge spans exactly one height level, so a
    hidden state can no longer jump levels. Shared nodes stay shared (this
    transform isolates the level-jump axis; combine with to_single_parent
    for the full degradation). Returns a NEW tree.
    """
    memo = {}
    rebuilt = {}

    def walk(node):
        nid = id(node)
        if nid in rebuilt:
            return rebuilt[nid]
        clone = NeuroNode(node.x, node.t_d, **_fields(node))
        _copy_meta(clone, node)
        rebuilt[nid] = clone
        h_parent = _height(node, memo)
        for child in node.C:
            new_child = walk(child)
            gap = h_parent - _height(child, memo)
            for _ in range(gap - 1):  # pad so each edge spans one level
                new_child = NeuroNode(None, None, C=[new_child])
            clone.insert(new_child)
        return clone

    return walk(root)


ABLATIONS = {
    'none': lambda t: t,
    'single-parent': to_single_parent,
    'no-level-jump': insert_level_padding,
    'both': lambda t: insert_level_padding(to_single_parent(t)),
}


def count_nodes(root):
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node.C)
    return len(seen)

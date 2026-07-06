import torch

from aan.constant.common_keywords import \
    CHILD_COUNT_KEY, CHILD_ADJACENCY_MATRIX_KEY, \
    INDICES_KEY, \
    HIDDEN_KEY, X_KEY, Y_KEY, \
    DECONV_HIDDEN_KEY, DECONV_X_KEY, \
    PREDICT_KEY, SUBTASK_KEY, VISIT_COUNT_KEY


class BatchNeuroTree(object):
    """A mini-batch of neuronodes processed together at one DFC/DFD step.

    This is a light view over a list of NeuroNode objects. Getters return
    plain lists aligned with ``nodes``; setters write attributes back onto
    the nodes so parents can reuse already-computed hidden states
    (multiple-parent structure, paper Algorithm 4).
    """

    def __init__(self, treeset):
        self.nodes = list(treeset)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    # ---------- generic access ----------

    def get(self, name):
        return [node.__dict__[name] for node in self.nodes]

    def get_i(self, name, i):
        return self.nodes[i].__dict__[name]

    def get_with_indices(self, name, indices):
        return [self.get_i(name, i) for i in indices]

    def set(self, name, batch_item):
        for i in range(len(self.nodes)):
            self.set_i(name, batch_item[i], i)

    def set_i(self, name, item, i):
        self.nodes[i].__dict__[name] = item

    def set_with_indices(self, name, items, indices):
        for item, idx in zip(items, indices):
            self.set_i(name, item, idx)

    # ---------- named accessors ----------

    def getVisitCount(self):
        return self.get(VISIT_COUNT_KEY)

    def getChildCount(self):
        return self.get(CHILD_COUNT_KEY)

    def getChildAdjacencyMatrix(self):
        return self.get(CHILD_ADJACENCY_MATRIX_KEY)

    def getIndices(self):
        return self.get(INDICES_KEY)

    def getX(self):
        return self.get(X_KEY)

    def getY(self):
        return self.get(Y_KEY)

    def getDeconvX(self):
        return self.get(DECONV_X_KEY)

    def getHiddens(self):
        return self.get(HIDDEN_KEY)

    def getDeconvHiddens(self):
        return self.get(DECONV_HIDDEN_KEY)

    def getPredicts(self):
        return self.get(PREDICT_KEY)

    def getSubTasks(self):
        return self.get(SUBTASK_KEY)

    def getSubtask(self, i):
        return self.get_i(SUBTASK_KEY, i)

    def setHiddens(self, batch_item):
        self.set(HIDDEN_KEY, batch_item)

    def setDeconvHiddens(self, batch_item):
        self.set(DECONV_HIDDEN_KEY, batch_item)

    def setDeconvXs(self, batch_item):
        self.set(DECONV_X_KEY, batch_item)

    def setIndices(self, batch_item):
        self.set(INDICES_KEY, batch_item)

    def setPredicts(self, batch_item):
        self.set(PREDICT_KEY, batch_item)

    def set_xh(self, batch_x, batch_h):
        """Accumulate deconv inputs delivered from (possibly many) parents."""
        for i, node in enumerate(self.nodes):
            if node.dx is None:
                node.dx = batch_x[i]
                node.dh = batch_h[i]
            else:
                node.dx = node.dx + batch_x[i]
                node.dh = node.dh + batch_h[i]

    # ---------- children ----------

    def get_child_count(self):
        return max(self.getChildCount())

    def get_child(self, number):
        """Children at position ``number``, with the batch indices that have one."""
        has_child_indices = []
        batch_child = []
        for i, node in enumerate(self.nodes):
            if len(node.C) > number:
                batch_child.append(node.C[number])
                has_child_indices.append(i)
        return BatchNeuroTree(batch_child), has_child_indices

    def get_child_i(self, number, i):
        return self.nodes[i].C[number]

    def get_child_hiddens(self, zero_vector):
        """Stack children's stored hidden states into (batch, max_children, H).

        Missing child positions are padded with ``zero_vector``.
        """
        max_child_count = self.get_child_count()
        if max_child_count == 0:
            return None
        if max_child_count == 1:
            # chain-shaped level: one stack for the whole batch instead of
            # one stack per node (thousands of small CUDA kernels per epoch)
            hiddens = [node.C[0].h if node.C else zero_vector for node in self.nodes]
            return torch.stack(hiddens, dim=0).unsqueeze(1)
        batch_hiddens = []
        for node in self.nodes:
            hiddens = [child.h for child in node.C]
            hiddens.extend([zero_vector] * (max_child_count - len(hiddens)))
            batch_hiddens.append(torch.stack(hiddens, dim=0))
        return torch.stack(batch_hiddens, dim=0)

    # ---------- visit bookkeeping (DFC / DFD, paper Algorithms 4, 5) ----------

    def visit(self):
        for node in self.nodes:
            node.count += 1

    def dvisit(self):
        for node in self.nodes:
            node.discount += 1

    def get_no_calculated(self):
        """Nodes visited for the first time in this DFC pass (compute once)."""
        return BatchNeuroTree([node for node in self.nodes if node.count <= 1])

    def get_calculated(self):
        return BatchNeuroTree([node for node in self.nodes if node.count == node.discount])

    def get_final_visit(self):
        """Nodes that have now received messages from all of their parents."""
        return BatchNeuroTree([node for node in self.nodes if node.count == node.discount])

    # ---------- loss bookkeeping ----------

    def init_sub_loss_count(self):
        for node in self.nodes:
            node.sub_loss_count = 0

    def init_main_loss_count(self):
        for node in self.nodes:
            node.main_loss_count = 0

    def get_nodes_to_calculate_sub_loss(self):
        """Nodes whose subtask loss has not been counted yet in this pass."""
        selected = []
        for node in self.nodes:
            node.sub_loss_count += 1
            if node.sub_loss_count == 1:
                selected.append(node)
        return BatchNeuroTree(selected)

    def get_nodes_to_calculate_main_loss(self):
        selected = []
        for node in self.nodes:
            node.main_loss_count += 1
            if node.main_loss_count == 1:
                selected.append(node)
        return BatchNeuroTree(selected)

    def get_task_nodes(self, name):
        return BatchNeuroTree([node for node in self.nodes if node.s_t == name])

    # ---------- batch (dis)assembly for sparse child positions ----------

    def batch_assemble(self, items, indices, zero_hidden):
        """Scatter ``items`` back to full batch size, padding absent rows."""
        index_set = set(indices)
        result = []
        point = 0
        for i in range(len(self.nodes)):
            if i in index_set:
                result.append(items[point])
                point += 1
            else:
                result.append(zero_hidden)
        return torch.stack(result, dim=0)

    def batch_disassemble(self, items, indices):
        return torch.stack([items[i] for i in indices], dim=0)

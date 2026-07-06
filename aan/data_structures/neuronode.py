class NeuroNode(object):
    """A node of a neurotree (paper, Section III-B).

    A neuronode NN = {x, t_d, s_t, A_c, C} carries arbitrary input data,
    its domain, an optional node-level subtask, the adjacency matrix among
    its children, and the children themselves. A node may be inserted as a
    child of several parents (multiple-parent structure), as long as the
    descendants of a node never include its ancestors.
    """

    def __lt__(self, other):
        return True

    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, x, t_d, s_t=None, A_c=None, C=None, E_c=None):
        self.x = x      # input data (any domain)
        self.t_d = t_d  # domain of the data
        self.s_t = s_t  # node-level subtask name (None: no subtask)
        self.A_c = A_c  # adjacency matrix among children (None: identity)
        self.E_c = E_c  # multi-dimensional edge features among children

        self.C = []
        self.C_count = 0
        if C is not None:
            for c in C:
                self.insert(c)

        self.pred = None
        self.y = None
        self.label = None

        self._init_state()

    def _init_state(self):
        """Initialize this node's per-forward-pass state (this node only)."""
        self.count = 0            # DFC visit count
        self.discount = 0         # DFD visit count
        self.sub_loss_count = 0
        self.main_loss_count = 0
        self.h = None             # hidden state stored by DFC
        self.dx = None            # deconv input accumulator (DFD)
        self.dh = None            # deconv hidden accumulator (DFD)
        self.indices = None       # maxpool indices, reused for unpooling
        self.child_h = []

    def reset_state(self):
        """Clear per-forward-pass state recursively so the tree can be reused.

        Multiple parents may share a node, so the traversal is cycle-safe
        via an identity-visited set.
        """
        visited = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            node._init_state()
            stack.extend(node.C)

    def insert(self, node):
        self.C.append(node)
        self.C_count += 1

    def dfs_travel_post_order_left_first(self, name=None, lv=0):
        """Print nodes in DFC (convolution) order."""
        if name is not None:
            print(name)
        for child in self.C:
            child.dfs_travel_post_order_left_first(lv=lv + 1)
        print(self.x, self.t_d, self.s_t, lv)

    def dfs_travel_pre_order_right_first(self):
        """Print nodes in DFD (deconvolution) order."""
        print(self.x, self.t_d, self.s_t)
        for child in reversed(self.C):
            child.dfs_travel_pre_order_right_first()

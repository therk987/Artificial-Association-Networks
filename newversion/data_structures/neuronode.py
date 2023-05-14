class NeuroNode(object):
    # Data Structure
    # the init args is required (x, t, A_c, C)
    def __lt__(self, other):
        return True

    def __getitem__(self,key):
        return getattr(self, key)

    def __init__(self, x, t_d, s_t=None, A_c = None, C = [], E_c = None):
        ####### Required #######
        self.x = x  # data
        self.t_d = t_d  # type of data
        self.s_t = s_t  # info of task
        self.A_c = A_c  # Children Adjacency matrix
        self.E_c = E_c  # Children Edge Features
        # why we defined the 'children' graph?
        # because it is to learn from a typical tree data structure without the cost of converting to GT.
        self.count = 0
        self.discount = 0
        self.sub_loss_count = 0
        self.main_loss_count = 0

        self.pred = None
        self.y = None
        self.label = None
        self.is_calculated = False

        self.C = []
        self.C_count = 0

        if C is not None:
            for c in C:
                self.insert(c)

        self.child_h = []

        self.h = None # already conv?
        self.dx = None
        self.dh = None
        ####### Required #######



        ####### FOR AutoEncoder #######
        self.indices = None  # when we used the MaxPooling to aggregate vectors, we can store the indices of the maxpool
        ####### FOR AutoEncoder #######


    def insert(self, tree):
        self.C.append(tree)
        if type(self.C_count) == int:
            self.C_count += 1

    def dfs_travel_post_order_left_first(self, name = None, lv = 0):  ## pre-view the convolution path
        if name is not None :
            print(name)
        node_count = len(self.C)
        for i in range(node_count):
            self.C[i].dfs_travel_post_order_left_first(name, lv = lv + 1)
            ## i => leaf-first
        print(self.x, self.t_d, self.s_t, lv)  ## post-order

    def dfs_travel_pre_order_right_first(self):  ## pre-view the deconvolution path
        print(self.x, self.t_d, self.s_t)  ## pre-order
        node_count = len(self.C)
        for i in range(node_count):
            self.C[node_count - i - 1].dfs_travel_pre_order_right_first()
            ## node_count - i - 1 => right-first

    def join(self, treeset):

        keys = set(self.__dict__.keys())
        keys.update(self.__dict__.keys())
        for tree in treeset:
            keys.update(tree.__dict__.keys())
            for key in self.__dict__.keys():
                # print(key)
                if self.__dict__[key] is not None and tree.__dict__[key] is not None:
                    self.__dict__[key].extend(tree.__dict__[key])

                else:
                    if self.__dict__[key] is None and tree.__dict__[key] is None:
                        pass

                    elif self.__dict__[key] is None:
                        self.__dict__[key] = [None for _ in range(len(self.x))]
                        self.__dict__[key].extend(tree.__dict__[key])
                    else:
                        tree.__dict__[key] = [None for _ in range(len(tree.x))]
                        self.__dict__[key].extend(tree.__dict__[key])




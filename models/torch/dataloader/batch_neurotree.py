import torch
import functools
class BatchNeuroTree(object):
    # Batch Association to merge AT Datas
    def __init__(self, treeset):
        self.nodes = treeset
        self.cache = {}
        self.iter_i = 0

    def __iter__(self):
        return self  # 현재 인스턴스를 반환

    def __len__(self):
        return len(self.nodes)

    def __next__(self):
        if self.iter_i < len(self):  # 현재 숫자가 반복을 끝낼 숫자보다 작을 때
            r = self.iter_i  # 반환할 숫자를 변수에 저장
            self.iter_i += 1  # 현재 숫자를 1 증가시킴
            return self.nodes[r]  # 숫자를 반환
        else:  # 현재 숫자가 반복을 끝낼 숫자보다 크거나 같을 때
            raise StopIteration  # 예외 발생

    def initial(self):

        keys = set(self.nodes[0].__dict__.keys())
        for tree in self.nodes:
            keys.update(tree.__dict__.keys())

        for key in keys:
            items = []
            for tree in self.nodes:
                if key in tree.__dict__.keys():
                    items.append(tree.__dict__[key])
                else:
                    items.append(None)

            self.cache[key] = items

    @functools.lru_cache(maxsize=3)
    def get(self, name):
        return [_.__dict__[name] for _ in self.nodes]

    ########## Version 2 ##########

    # @functools.lru_cache(maxsize=3)
    # def get_child_hiddens(self):
    #     batch_child_hidden = []
    #     for i, node in enumerate(self.nodes):
    #         node_hiddens = []
    #         for j, child in enumerate(node.C):
    #             node_hiddens.append(child.h)  # number에 해당하는 자식을 꺼내옴!
    #
    #         if len(node_hiddens) == 0:
    #             node_hiddens =
    #         node_hiddens = torch.stack(node_hiddens, dim = 0)
    #         batch_child_hidden.append(node_hiddens)
    #     return batch_child_hidden

    ########## Version 2 ##########

    ########## Version 1 ##########
    @functools.lru_cache(maxsize=3)
    def get_child_hidden(self, number, zero_vector):
        batch_child_hidden = []
        for i, node in enumerate(self.nodes):
            if len(node.C) < number + 1:  # number 에 해당하는 자식이 없음!
                batch_child_hidden.append(zero_vector)
            else:
                batch_child_hidden.append(node.C[number].h)  # number에 해당하는 자식을 꺼내옴!
                # print(node.C[number])
                # print(node.C[number].h)

        return torch.stack(batch_child_hidden)

    # @functools.lru_cache(maxsize=3)
    def get_child_hiddens(self, zero_vector):
        max_child_count = self.get_child_count()
        result = []
        if max_child_count == 0:
            return None
        for i in range(max_child_count):
            batch_child_i_hidden = self.get_child_hidden(i, zero_vector)
            result.append(batch_child_i_hidden)
        return torch.stack(result, dim=1)


########## Version 1 ##########

    # @functools.lru_cache(maxsize=1)
    def get_i(self, name, i):
        return self.nodes[i].__dict__[name]

    # @functools.lru_cache(maxsize=1)
    def get_with_indices(self, name, indices):
        return [self.get_i(name, _) for _ in indices]

    def set(self, name, batch_item):
        for i in range(len(self.nodes)):
            self.set_i(name, batch_item[i], i)

    def set_xh(self, batch_x, batch_h):
        for i in range(len(self.nodes)):
            if self.nodes[i].dx == None:
                self.nodes[i].dx = batch_x[i]
                self.nodes[i].dh = batch_h[i]
            else:
                self.nodes[i].dx = self.nodes[i].dx + batch_x[i]
                self.nodes[i].dh = self.nodes[i].dh + batch_h[i]

    def set_i(self, name, item, i):
        self.nodes[i].__dict__[name] = item

    def batch_set_with_indices(self, name, items, indices):
        for item, idx in zip(items, indices):
            self.set_i(name, item, indices[idx])

    def get_task_nodes(self, name):
        batch_task_nodes = []
        for i, node in enumerate(self.nodes):
            if node.t_t == name:
                batch_task_nodes.append(node)
        return BatchNeuroTree(batch_task_nodes)
    @functools.lru_cache(maxsize=3)
    def get_child_count(self):
        return max(self.get('C_count'))

    @functools.lru_cache(maxsize=3)
    def get_child(self, number):
        has_child_indices = []
        batch_child = []
        for i, node in enumerate(self.nodes):
            if len(node.C) < number + 1:  # number 에 해당하는 자식이 없음!
                pass
            else:
                batch_child.append(node.C[number])  # number에 해당하는 자식을 꺼내옴!
                has_child_indices.append(i)  # i 번째 노드에는 number에 해당하는 자식이 있다!
        return BatchNeuroTree(batch_child), has_child_indices

    def visit(self):
        for i, node in enumerate(self.nodes):
            node.count += 1

    def dvisit(self):
        for i, node in enumerate(self.nodes):
            node.discount += 1

    def get_final_visit(self):
        final_visits = []
        for i, node in enumerate(self.nodes):
            if node.count == node.discount :  # number 에 해당하는 자식이 없음!
                final_visits.append(node)  # i 번째 노드에는 number에 해당하는 자식이 있다!
        return BatchNeuroTree(final_visits)
    # @functools.lru_cache(maxsize=3)

    def init_sub_loss_count(self):
        for i, node in enumerate(self.nodes):
            node.sub_loss_count = 0

    def init_main_loss_count(self):
        for i, node in enumerate(self.nodes):
            node.main_loss_count = 0


    def get_nodes_to_calculate_sub_loss(self): # 계산되지 않은 애들만!
        loss_calculated_nodes = []
        for i, node in enumerate(self.nodes):
            node.sub_loss_count += 1
            if node.sub_loss_count > 1:  # number 에 해당하는 자식이 없음!
                pass
            else:
                loss_calculated_nodes.append(node)  # i 번째 노드에는 number에 해당하는 자식이 있다!

        return BatchNeuroTree(loss_calculated_nodes)

    def get_nodes_to_calculate_main_loss(self): # 계산되지 않은 애들만!
        loss_calculated_nodes = []
        for i, node in enumerate(self.nodes):
            node.main_loss_count += 1
            if node.main_loss_count > 1:  # number 에 해당하는 자식이 없음!
                pass
            else:
                loss_calculated_nodes.append(node)  # i 번째 노드에는 number에 해당하는 자식이 있다!

        return BatchNeuroTree(loss_calculated_nodes)



    def get_no_calculated(self):
        no_calculated_nodes = []
        for i, node in enumerate(self.nodes):
            if node.count > 1:  # number 에 해당하는 자식이 없음!
                pass
            else:
                no_calculated_nodes.append(node)  # i 번째 노드에는 number에 해당하는 자식이 있다!
        return BatchNeuroTree(no_calculated_nodes)

    def get_calculated(self):
        calculated_nodes = []
        for i, node in enumerate(self.nodes):
            if node.count == node.discount:  # number 에 해당하는 자식이 없음!
                calculated_nodes.append(node)  # i 번째 노드에는 number에 해당하는 자식이 있다!
        return BatchNeuroTree(calculated_nodes)

    @functools.lru_cache(maxsize=3)
    def get_child_i(self, number, i):
        return self.nodes[i].C[number]

    def batch_assemble(self, items, indices, zero_hidden):
        batch_count = len(self.nodes)
        result = []
        tmp_indices = indices[:]
        point = 0
        for i in range(batch_count):
            if i in tmp_indices:
                result.append(items[point])
                point += 1
            else:
                result.append(zero_hidden)
        return torch.stack(result, dim=0)

    def batch_disassemble(self, items, indices):
        return torch.stack([items[_] for _ in indices], dim=0)

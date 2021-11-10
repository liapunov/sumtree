# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 22:19:51 2021

@author: frina
"""
import numpy as np
from numpy.random import randint

# Thanks to AdventuresinML for inspiring the implementation
# and providing some useful snippet (in particular the code for the
# generation of trees from a list is taken from the link below)
# https://adventuresinmachinelearning.com/sumtree-introduction-python/

class STNode():
    def __init__(self, left=None, right=None, is_leaf=False,
                 is_new_root=False, val=0, payload=None):
        '''Define structure and behavior of a SumTree node.'''
        
        assert (is_leaf == True and left==None and right==None)\
               or\
               (not is_leaf and payload==None),\
               "This node has an inconsistent state. is_leaf={is_leaf},\
val={val}, right={right} and left={left}."
        
        self.is_new_root = is_new_root
        self.is_leaf = is_leaf
        self.left_child = left
        self.right_child = right
        self.val = val
        self.payload = payload
        self.parent = None
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
        
    @classmethod
    def createLeaf(cls, val=0, payload=None):
        if payload is None:
            raise ValueError("Tried to insert a null value into the tree")
            
        leaf = cls(is_leaf=True, val=val, payload=payload)
        return leaf
    
class SumTree(object):
    '''Define an efficient data structure for the prioritized replay buffer.'''
    
    def __init__(self, root: STNode = None):
        if root is not None:
            self.root = root
        else:
            self.root = STNode(is_new_root=True)
    
    def insert_recur(self, new, cur):
        if cur.is_leaf:
            # we are at a leaf or the root is empty: need to extend the tree
            leaf_from_node = STNode.createLeaf(val=cur.val, 
                                               payload=cur.payload)
            cur.is_leaf = False
            cur.payload = None
            cur.left_child = new
            cur.right_child = leaf_from_node
            new.parent = cur
            leaf_from_node.parent = cur
            cur.val += new.val #internal node sum gets updated   
        else: # traversing an internal node
            cur.val += new.val
            if cur.left_child.val < cur.right_child.val:
                self.insert_recur(new, cur.left_child)
            else:
                self.insert_recur(new, cur.right_child)
    
    def insert(self, new=None, cur=None, val=0, payload=None):
        if new is None:
            new = STNode.createLeaf(val=val, payload=payload)
        if cur is None:
            cur = self.root
        if cur.is_new_root:
            self.root = new
            self.root.is_new_root = False
            return
        self.insert_recur(new, cur)
        
    def retrieve(self, val, cur=None):
        if cur is None:
            cur = self.root
        if cur.is_leaf:
            return cur
        else:
            l_val = cur.left_child.val
            if val <= l_val:
                return self.retrieve(val, cur.left_child)
            if val <= self.root.val:
                return self.retrieve(val - l_val, cur.right_child)
            else:
                raise ValueError("Trying to extract a value with priority \
higher than the maximum allowed")
        
    
    def update(self, node, value, payload=None):
        if node.is_leaf:
            diff = value - node.val
            node.val = value
            if payload is not None:
                node.payload = payload
            if node.parent is not None:
                self.update(node.parent, diff)
        else:
            node.val += value
            if node.parent is not None:
                self.update(node.parent, value)
    
    @classmethod
    def createFromList(list_of_pairs):
        # original code by adventuresinML, lightly adapted
        nodes = [STNode.createLeaf(val=v[0], payload=v[1]) 
                 for i, v in enumerate(list_of_pairs)]
        leaf_nodes = nodes
        while len(nodes) > 1:
            # note: iter is used as a clever way to pair
            # consecutive nodes (iter always yelds the next element)
            inodes = iter(nodes)
            nodes = [STNode(*pair) for pair in zip(inodes, inodes)]
        return nodes[0], leaf_nodes

class PriorityQueue(SumTree):
    
    def __init__(self, root=None, size=0, max_size=20000,
                 initial_list:list=None):
        if initial_list is None:
            super().__init__(root)
        else:
            super().__init__(root)
        self.size = size
        self.max_size = max_size
        self.w_max = 0
        self.alpha = 0.6 # used for importance sampling of the priority values
        self.beta = 0.4 # used for importance sampling of the priority values
        self.beta_rate = 1.000001 # growth factor for beta
    
    def _calcImportance(self, metric):
        """Adjust the weights so that in the long run they do not skew
        the model distribution.
        
        Using the adjustment found in https://arxiv.org/abs/1511.05952v4
        That is,
        w_{i} = \left(\frac{1}{N}\cdot\frac{1}{P\left(i\right)}\right)^{\beta} 
        """
        if metric is None:
            return 1000
        epsilon = 0.0001
        adj_pi = (np.abs(metric) + epsilon) ** self.alpha
        total_pi = self.root.val + epsilon
        imp_sampled_pi = (total_pi / ((self.size + 1) * adj_pi)) ** self.beta
        # beta = 0.5 maxes out at 1 after around
        # 100.000 cicles with a beta rate of 1.000001
        # maxing out is bad (infinite recursion)
        self.beta = min(self.beta*self.beta_rate, 0.999)
        return imp_sampled_pi*adj_pi
    
    def get_size(self):
        return self.size
    
    def insert(self, new=None, cur=None, val=0, payload=None):
        # First check whether we reached the limit
        if new is None:
            priority = self._calcImportance(val)
            new = STNode.createLeaf(val=priority, payload=payload)
        if self.size >= self.max_size:
            # if we have reached capacity, replace a value
            # (no need to insert: find a similar value and replace it)
            to_replace = self.retrieve(new.val)
            to_replace.val = new.val
            to_replace.payload = new.payload
            return
        else:
            self.size += 1
        super().insert(new)
    
    def retrieve(self, val, from_node=None):
        return super().retrieve(val, from_node)
    
    def sample(self, sample_size=1000):
        # the root of the sumtree has value equal to the total
        total_sum = self.root.val
        selected = [self.retrieve(randint(0, total_sum)).payload
                    for _ in range(sample_size)]
        return selected
    
    def batchInsert(self, vals, payloads):
        assert len(vals) <= self.max_size, f"a maximum buffer size of \
{self.max_size} was specified, \
but the list provided has length {len(vals)}."
        assert len(vals) == len(payloads),\
        "values and payloads have different lengths."
        for val, payload in zip(vals, payloads):
            self.insert(val=val, payload=payload)
        
    @classmethod
    def createFromList(cls, buffer_list, max_size):
        
        size = len(buffer_list)
        tree = SumTree.createFromList(buffer_list)
        tree_root = tree.root
        del tree
        return cls(root=tree_root, size=size, max_size=max_size)
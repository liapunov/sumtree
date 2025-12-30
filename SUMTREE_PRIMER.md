# SumTree Introduction (primer)
_Markdown export of `src/SumTree Introduction.ipynb`. For the executable version with charts, open the notebook directly._

```python
from numpy.random import random, randint, choice
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from time import time
```

```python
from Sumtree import STNode, SumTree, PriorityQueue
```


# Priority Queues and Sum Trees

A priority queue is a queue where elements with high priority are extracted before lower-priority items. A sum tree acts like a stochastic priority queue: elements with higher priority still have a better probability of being sampled first, but every element retains a non-negligible chance of being drawn.

## Why stochastic sampling with priority is useful

A priority queue—such as a Fibonacci heap—can support optimization and real-time scheduling. Deterministic extraction, however, may be undesirable when we need some exploration; we might want all elements to remain available even if their priority is lower.

The most relevant modern application is Reinforcement Learning. In prioritized experience replay (Schaul et al., 2015: https://paperswithcode.com/paper/prioritized-experience-replay), a sum tree can reprocess training cases by giving higher “surprise” examples a better chance to be replayed, while still letting the agent occasionally revisit the rest of the buffer.

## How sum trees work

Sum trees are binary trees where each internal node value is the sum of its children. They allow a binary search over a permuted cumulative distribution: we map a uniform random number to a weighted sample.

Sum trees do not actually operate on CDFs, but the sampling intuition is similar:
- With empirical CDFs as sorted arrays, you extract a random number $n$ in $[0,1]$ and pick the index of the value immediately greater than $n$.
- With sum trees, after extracting the random number, you walk down the tree according to how values are summed and pick the value in the final leaf node.

### How to pick up a leaf in a sum tree

For an existing sum tree, the rule to pick up a leaf value is straightforward:
1. Extract a random number in the interval $[0, root\_val)$.
2. At any internal node (starting from root), check if the number to look for ($n$) is greater or smaller than the left-child value.
   - If it is greater, subtract the value of the left child from $n$ and proceed to the right child.
   - If it is smaller, proceed to the left child.
3. When at a leaf, return the object associated with that leaf. This can be just the value itself, or any other payload.

How does this work for sampling?
- Each node value is the sum of the values of the children, so each node is the sum of two or more leaves, and each node is a root of a smaller sum tree.
- The value of $n$ cannot be greater than $root\_val$, the value of the root.
- When moving to the right, $n$ is reduced by $l\_val$, so it is now in the range $[0, r\_val]$.
- We can reason as if we were always dealing with the root at each iteration (and we will call the search number $n$, even if the number actually can get smaller by going down the tree).
- As we assume that $n$ is extracted randomly from a uniform distribution, if $l\_val$ is the value of the left child and $r\_val$ is the value of the right child, we will have probability $\frac{l\_val}{root\_val}$ to go through the left branch and probability $\frac{r\_val}{root\_val}$ to go down the right branch.
- Repeating the process iteratively to the leaf of a tree with $l$ levels, the probability to end up in a certain leaf with value $leaf\_val$ is

$\Large\frac{leaf\_val}{node\_val_{l-1}}\times\frac{node\_val_{l-1}}{node\_val_{l-2}}\times...\times\frac{node\_val_{2}}{root\_val} = \frac{leaf\_val}{root\_val}.$

Sum tree sampling therefore mirrors a weighted random selection where each leaf’s contribution is proportional to its value.


## A basic implementation: SumTree
The module `Sumtree.py` implements a `SumTree` with three core methods: `insert`, `retrieve`, and `update`, where `update` performs bottom-up value updates after a leaf changes. There is also a class method to create the tree from an existing list, `createFromList`, contributed by @adventuresinML; as explained, it is not essential for this implementation but helps with quick setup.

The first tree we will instantiate is quite small so that it can be printed. The `insert` method accepts either an existing node, or a couple of parameters:
- `val`, indicating the priority
- `payload`, the object that we want to sample from the tree

In order to visualize things better, in these examples the payload will be equal to the priority value.

```python
tree0 = SumTree()

# inserting a few values in the range [1, 500]
insert_times = []
vals = []
for i in range(20):
    val = randint(1,500)
    start_insert = time()
    tree0.insert(val=val, payload=val)
    end_insert = time()
    vals.append(val)
    insert_times.append(end_insert - start_insert)
    
sns.scatterplot(x=vals, y=insert_times)
```


As the scatter plot shows, most insertions take around 4–5 microseconds with no meaningful dependency on the magnitude of the number.

If we try to print out the tree, it seems pretty balanced. Keeping a sum tree balanced is relatively easy because each internal node tracks the sums of the children, so inserting into the lighter branch tends to spread values. There is no strict guarantee, though: a very large value can temporarily skew the tree until later insertions rebalance it.

```python
# Binary tree pretty print util
# Credits:
# https://stackoverflow.com/a/54074933 (original thread)
# https://stackoverflow.com/users/1143396/j-v (first creator)
# https://stackoverflow.com/users/4237254/bck (revision)
# (removed some branches we are not using here)
def print_tree(root, val="val", left="left_child", right="right_child"):
    def display(root, val="val", left="left_child", right="right_child"):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if getattr(root, right) is None and getattr(root, left) is None:
            line = '%s' % getattr(root, val)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Two children.
        left, n, p, x = display(getattr(root, left))
        right, m, q, y = display(getattr(root, right))
        s = '%s' % getattr(root, val)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    lines, *_ = display(root, val, left, right)
    for line in lines:
        print(line)
        
print_tree(tree0.root)
```

### Performance

Now that we have seen a sum tree in the wild, let's try to insert a significant number of items (100,000) and see what happens if we sample from the tree.

Here below, we show how SumTree works when we insert a big array of increasing values. The expectation is that higher values will be sampled more frequently.

```python
tree = SumTree()

# inserting 100000 values in the range [1, 100000].
# We are also measuring how long does it take for
# SumTree.insert to tackle increasingly large values.
insert_times = np.zeros(100000)
vals = []
for i in range(100000):
    val = random()
    vals.append(val)
    start_insert = time()
    tree.insert(val=val, payload=val)
    end_insert = time()
    insert_times[i] = end_insert - start_insert

indexes = []

# Sampling 100,000 values from the sumtrees.
# How long will it take?
rtval = tree.root.val
start = time()
for i in range(100000):
    random_val = random() * rtval 
    payload = tree.retrieve(random_val).payload
    indexes.append(payload)
end = time()

print(f"The insertions took {np.sum(insert_times)} and the sampling took {end-start} seconds.")

# drawing insert times and the distribution of sampled values
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.lineplot(x=list(range(100000)), y=insert_times, ax=ax[0])
sns.histplot(sorted(indexes), ax=ax[1])
```

```python
%timeit STNode.createLeaf(val=random(), payload=0)
```


Sampling 100,000 values from the tree took around 1 second, or 0.00001 seconds per sampling on average.

The two graphs above show respectively the times of each insertion and the distribution of the payloads, conveniently defined here as pure numbers (so that they can be ordered easily on the graph).

As one can see:
- Most samples are inserted in a very short time, though occasionally a few insertions are noticeably slower and outliers longer than one tenth of a second exist.
- With 100,000 samples the tree could follow almost perfectly the original distribution.

So, it seems that this sum tree does the job, although a small number of insertions may be catastrophically slower than the others. Since there are only three or four such outliers out of 100,000 insertions, we might ignore the problem for the moment; but we might need to investigate its causes (they could even be related to OS issues rather than the algorithm itself).

```python
large = np.where(insert_times>0.01)[0]
print(f"These values caused an outlandishly long insertion time: {[(i, vals[i]) for i in large]}")
```


In any case, we can check and make sure that the sum tree is reasonably well balanced. We cannot do it with a tree printing utility like before, as the tree is huge, but we can verify that the maximum and the average depth do not deviate much from the theoretical $log_2 n$.

```python
depths = []
def sumTreeCheck(node, level):
    """Check the consistency of the nodes and gauges the depths of each leaf."""
    if node.is_leaf:
        depths.append(level)
        return
    else:
        if node.left_child is not None and node.right_child is None:
            if round(node.val, 5) != round(node.left_child.val,5):
                print(f"found node with an inconsistent sum: the parent has sum\
 {node.val} but the left child has value {node.left_child.val}")
            sumTreeCheck(node.left_child, level+1)
        elif node.right_child is not None and node.left_child is None:
            if round(node.val, 5) != round(node.right_child.val, 5):
                print(f"found node with an inconsistent sum: the parent has sum\
 {node.val} but the left child has value {node.right_child.val}")
            sumTreeCheck(node.right_child, level+1)
        elif node.right_child is not None and node.left_child is not None:
            if round(node.val, 5) != round(node.right_child.val + node.left_child.val, 5):
                print(f"found node with an inconsistent sum: the parent has sum\
 {node.val} but the children have value {node.left_child.val} and {node.right_child.val}")
            sumTreeCheck(node.right_child, level+1)
            sumTreeCheck(node.left_child, level+1)

sumTreeCheck(tree.root, 0)
print(f"The maximum depth of the tree is {max(depths)} and the average depth is \
{np.mean(depths)} (standard deviation {np.std(depths)}),\n\
while the logarithm of the number of leaves ({len(depths)}) is {np.log2(len(depths))}")
```


# PriorityQueue, a sophisticated SumTree

So far, (almost) so good. In order to use a sum tree properly for Prioritized Experience Replay (PER), we might need some more pieces.
- Define a maximum size for the tree, and a policy for replacing leaves
- Modulate the sampling policy (e.g. importance sampling) - more on that later
- Update existing leaves (implemented in Sum Tree but unused)

We will go through the three points above in more detail now. `PriorityQueue` also includes a new method, `sample`, which accepts a `sample_size` parameter. 


## Importance sampling

According to Schaul et al. (https://arxiv.org/abs/1511.05952),
>The estimation of the expected value with stochastic updates relies on those updates corresponding
to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
converge to (even if the policy and state distribution are fixed).

For this reason, they introduced a modulating factor to the values used in the sum tree, which basically levels out the priorities at the high end of the distribution.

$w_{i} = \left(\frac{1}{N}\cdot\frac{1}{P\left(i\right)}\right)^{\beta}$, where $P\left(i\right) = \frac{p_{j}^{\alpha}}{\sum_{i}{p_{i}^{\alpha}}}$

The factor $w_{i}$ is then multiplied by the original priority before it is stored in the tree. The `PriorityQueue` implementation exposes the $\alpha$ and $\beta$ hyperparameters so they can be tuned.

This is a difference with the original application of IS described in Schaul, where it seems that the weighting is applied only during the parameter update, but since this is a monotone transformation that basically only "dampens" the effect of the sum tree to the highest priority payloads, and that the original motivation for applying importance sampling is to contain the effect of a linear prioritization according to value, it should not be a problem, and it simplifies the process.


## Performance

As one can see below, the performance of this more sophisticated version is slightly higher than the original `SumTree`, as the importance sampling step does take some milliseconds more. In the context of RL, where the insertion happens at the end of an agent play, it does not represent a significant difference.

```python
max_size = 100000
start = time()
prior = PriorityQueue(max_size=max_size)

insert_times_p = np.zeros(100000)
for i in range(100000):
    val = randint(1, 100000)
    start_insert = time()
    prior.insert(val=val, payload=val)
    end_insert = time()
    insert_times_p[i] = end_insert - start_insert

sample_start = time()
total = prior.root.val
sample = prior.sample(max_size)
sample_end = time()

print(f"A sample of {max_size} was generated in {sample_end - sample_start} seconds.")

# drawing insert times and the distribution of sampled values
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.lineplot(x=list(range(100000)), y=insert_times_p, ax=ax[0])
sns.histplot(sorted(sample), ax=ax[1])
```


Also, we expect the insertion of $n$ items to have complexity proportional to $n*log(n)$. This is verified below: the empirical timings track the reference $n\log n$ curves closely.

```python
insert_times = []
sample_times = []

for size in range(10000, 100001, 2000):
    print(f"testing size {size}...")
    prior2 = PriorityQueue(max_size=size)
    insert_start = time()
    for i in range(size):
        val = randint(1, size)
        prior2.insert(val=val, payload=val)
    insert_end = time()
    total = prior2.root.val
    sample = prior2.sample(size)
    sample_end = time()
    insert_times.append(insert_end - insert_start)
    sample_times.append(sample_end - insert_end)
```

```python
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
logline_x = np.linspace(0, 100000, 1000)
insert_logline = logline_x * np.log(logline_x) * (1/400000)
sns.lineplot(x=list(range(0, 100000, 100)), y=insert_logline, ax=ax[0])
sns.lineplot(x=list(range(10000, 100001, 2000)), y=insert_times, ax=ax[0])
sample_logline = logline_x * np.log(logline_x) * (1/600000)
sns.lineplot(x=list(range(0, 100000, 100)), y=sample_logline, ax=ax[1])
sns.lineplot(x=list(range(10000, 100001, 2000)), y=sample_times, ax=ax[1])
plt.show()
```


## Conclusion and next steps

This notebook has focused on a useful data structure for Reinforcement Learning that not many talk about in depth. Sum trees make for excellent priority buffers, but the insertion phase has a few subtleties and requires deliberate design choices. In any case, the common feature is a logarithmic complexity in both insertion and extraction, which justifies their use compared to normal arrays or other samplers.

The next update for this project would be to have this structure utilised in an RL method in an openai-gym environment.
Stay tuned.

## Appendix 1 - Can probability distributions be reconstructed by a Sum Tree?

```python
def ecdfFromDistro(distro):
    distro.insert(0, 0)
    for i in range(1, len(distro)):
        distro[i] = distro[i] + distro[i-1]
    return distro[1:]

def ecdfFromTree(tree):
    root = tree.root
    total = root.val
    distro = [0]
    def traverseAppend(node):
        if node.is_leaf:
            distro.append(node.val/total)
        else:
            traverseAppend(node.left_child)
            traverseAppend(node.right_child)
    traverseAppend(root)
    ecdf = ecdfFromDistro(distro)
    return ecdf

def drawECDF(tree):
    ecdf = ecdfFromTree(tree)
    x = list(range(len(ecdf)))
    sns.lineplot(x=x, y=ecdf, drawstyle='steps-pre')
    plt.show()
```

```python
drawECDF(prior)
```

```python
from scipy.stats import norm
x = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100000)
y = norm.pdf(x)
```

```python
sns.lineplot(x=x, y=y)
```

```python
tree_norm = SumTree()
for i in range(100):
    tree_norm.insert(val=y[i], payload=i)
    
drawECDF(tree_norm)
```

```python
prior_norm = PriorityQueue(max_size=100000)
for i in range(100000):
    prior_norm.insert(val=y[i], payload=i)
    
drawECDF(prior_norm)
```

```python
sample_norm = prior_norm.sample(10000)
sns.histplot(sample_norm)
```

```python
prior2 = PriorityQueue(max_size=max_size)
%timeit randint(1, max_size)
```

```python
print(insert_times)
print(sample_times)
```


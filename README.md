# sumtree
A small primer on sumtrees

I encountered sum trees first when reading about DQN and the concept of priority buffers. Before that, I had not encountered sum trees despite being a pretty simple form of binary tree. This repository is an exploration of sum trees with a personal implementation of both vanilla sum trees and sum-tree-based priority buffers for use in RL.

## Visual simulator

You can explore the sumtree interactively with the Streamlit dashboard.

### How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit will print a local URL (usually <http://localhost:8501>) where you can open the dashboard.

### How to use the dashboard

* **Configure the tree (sidebar).** Pick a random seed, number of leaves, and a distribution for leaf values (uniform, normal, exponential). Changing any of these regenerates the tree with new values; each leaf’s payload is its numeric value.
* **Inspect metrics.** The top row shows total nodes, leaves, maximum depth, min/max values, and the sum of all values so you can verify the tree’s structure at a glance.
* **View distributions.** Two charts show the raw leaf values (bar by leaf index) and a histogram of those values. These help you confirm the shape of the underlying distribution you generated.
* **Interactive sampling.** Use the “Samples per draw” slider and click **Draw samples** to retrieve weighted random values from the sumtree. Each draw appends samples to a live histogram so you can watch the sampled distribution converge toward the leaf-value distribution; **Reset histogram** clears the history.

Things to check: toggle between distributions to compare how the histograms change; increase the sample count to see faster convergence; and vary the leaf count to observe how the sumtree depth and metrics respond.

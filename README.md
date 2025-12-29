# sumtree
A small primer on sumtrees

I encountered sum trees first when reading about DQN and the concept of priority buffers. Before that, I had not encountered sum trees despite being a pretty simple form of binary tree. This repository is an exploration of sum trees with a personal implementation of both vanilla sum trees and sum-tree-based priority buffers for use in RL.

## Visual simulator

You can explore the sumtree interactively with the Streamlit dashboard:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app lets you configure the distribution of leaf values, inspect key metrics (node count, depth, min/max values), visualize the underlying distribution, and draw weighted samples while watching the histogram converge.

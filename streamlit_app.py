"""Interactive Streamlit dashboard for exploring sumtrees."""
import pathlib
import random
from typing import Iterable, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Allow importing the project module without packaging it
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in st.session_state.get("_path_setup", set()):
    # Avoid appending multiple times across reruns
    st.session_state.setdefault("_path_setup", set()).add(str(SRC_DIR))
    import sys
    sys.path.append(str(SRC_DIR))

from Sumtree import STNode, SumTree  # noqa: E402


def build_sumtree(values: Iterable[float]) -> SumTree:
    """Create a SumTree populated with the provided values."""
    tree = SumTree()
    for idx, val in enumerate(values):
        tree.insert(val=float(val), payload=float(val))
    return tree


def iter_leaves(node: STNode) -> Iterable[STNode]:
    if node is None:
        return
    if node.is_leaf:
        yield node
    else:
        yield from iter_leaves(node.left_child)
        yield from iter_leaves(node.right_child)


def count_nodes(node: STNode) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left_child) + count_nodes(node.right_child)


def max_depth(node: STNode, depth: int = 1) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return depth
    return max(
        max_depth(node.left_child, depth + 1),
        max_depth(node.right_child, depth + 1),
    )


def leaf_values(node: STNode) -> List[float]:
    return [leaf.val for leaf in iter_leaves(node)]


def sample_from_tree(tree: SumTree, draws: int) -> List[float]:
    total = tree.root.val
    return [
        tree.retrieve(random.uniform(0, total)).payload
        for _ in range(draws)
    ]


def render_metrics(tree: SumTree):
    leaves = list(iter_leaves(tree.root))
    values = [leaf.val for leaf in leaves]
    metrics = {
        "Total nodes": count_nodes(tree.root),
        "Leaves": len(leaves),
        "Max depth": max_depth(tree.root),
        "Max value": max(values),
        "Min value": min(values),
        "Sum of values": tree.root.val,
    }
    cols = st.columns(len(metrics))
    for col, (label, metric) in zip(cols, metrics.items()):
        col.metric(label, f"{metric:.2f}" if isinstance(metric, float) else metric)


def render_leaf_distribution(tree: SumTree):
    values = leaf_values(tree.root)
    df = pd.DataFrame({"value": values, "leaf": range(1, len(values) + 1)})
    st.subheader("Leaf value distribution")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("leaf:O", title="Leaf index"),
        y=alt.Y("value:Q", title="Value"),
        tooltip=["leaf", "value"],
    )
    st.altair_chart(chart, use_container_width=True)

    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title="Value bins"),
        y=alt.Y("count()", title="Count"),
    )
    st.altair_chart(hist, use_container_width=True)


def render_sampling_panel(tree: SumTree):
    st.subheader("Interactive sampling")
    st.write(
        "Draw random numbers weighted by their value and watch the histogram "
        "approach the underlying distribution."
    )

    if "sample_history" not in st.session_state:
        st.session_state["sample_history"] = []

    sample_count = st.slider("Samples per draw", min_value=1, max_value=500, value=50)
    if st.button("Draw samples"):
        st.session_state["sample_history"].extend(sample_from_tree(tree, sample_count))

    if st.button("Reset histogram"):
        st.session_state["sample_history"] = []

    history = st.session_state["sample_history"]
    if history:
        hist_df = pd.DataFrame({"sample": history})
        hist_chart = alt.Chart(hist_df).mark_bar().encode(
            x=alt.X("sample:Q", bin=alt.Bin(maxbins=30), title="Sampled value"),
            y=alt.Y("count()", title="Frequency"),
        )
        st.altair_chart(hist_chart, use_container_width=True)
        st.caption(
            "The histogram updates each time you draw; use the reset button to start over."
        )
    else:
        st.info("No samples yet. Draw some numbers to start the histogram.")


def main():
    st.title("SumTree visual simulator")
    st.write(
        "Experiment with a sumtree by generating weighted leaves, exploring their "
        "distribution, and interactively sampling from the tree."
    )

    with st.sidebar:
        st.header("Tree configuration")
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)
        leaf_count = st.slider("Number of leaves", min_value=5, max_value=200, value=50)
        distribution = st.selectbox(
            "Value distribution",
            ["uniform", "normal", "exponential"],
            format_func=lambda x: x.capitalize(),
        )
        st.caption(
            "Change the distribution or seed to generate a new tree. The payload of each "
            "leaf is its numeric value."
        )

    rng = np.random.default_rng(int(seed))
    if distribution == "uniform":
        values = rng.uniform(0.1, 1.0, size=leaf_count)
    elif distribution == "normal":
        values = np.abs(rng.normal(loc=0.5, scale=0.2, size=leaf_count)) + 0.05
    else:  # exponential
        values = rng.exponential(scale=0.4, size=leaf_count) + 0.05

    tree = build_sumtree(values)
    render_metrics(tree)
    render_leaf_distribution(tree)
    render_sampling_panel(tree)


if __name__ == "__main__":
    main()

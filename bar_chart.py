#%%
import matplotlib.pyplot as plt


def make_bar_chart(output_path=None):
    labels = ["Total", "Plume detected", "Sorted manually", "Wind filtered", "Passenger ferry"]
    values = [8000, 1800, 820, 620, 2000]
    colors = ["red", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylabel("#ship passes", fontsize=16)

    # increase tick label sizes and rotate x labels
    ax.tick_params(axis='x', labelrotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # annotate values above bars (larger text)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -18),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    labels = ["Manually sorted data", "Counted multiple times", "Counted only once", "Dreging ship", "Container ship 1", "Container ship 2"]
    values = [820, 480, 200, 13, 8, 7]
    colors = ["red", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylabel("#ship passes", fontsize=16)

    # increase tick label sizes and rotate x labels
    ax.tick_params(axis='x', labelrotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # annotate values above bars (larger text)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -18),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    make_bar_chart()

# %%

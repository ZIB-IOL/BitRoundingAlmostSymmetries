import random
import sys
from typing import List, Tuple

# Distributes total capacity into bins randomly, such that the bins sum up to the total capacity
def random_partition(total, bins):
    cuts = sorted(random.sample(range(1, total), bins - 1))
    cuts = [0] + cuts + [total]
    return [cuts[i+1] - cuts[i] for i in range(bins)]

# number of items
n = 1000

# number of bins
m = 50

# allow overriding defaults via CLI key=value arguments
_type_map = {
    "n": int,
    "m": int,
    "k": int,              # number of clusters
    "w_min": int,          # min cluster weight
    "w_max": int,          # max cluster weight
    "v_min": int,          # min base value per cluster
    "v_max": int,          # max base value per cluster
    "noise": int,          # uniform value noise range (+/-)
    "balanced_clusters": int,  # 1 for nearly equal cluster sizes, 0 for random
}

for _arg in sys.argv[1:]:
    if "=" not in _arg:
        continue
    _key, _value = _arg.split("=", 1)
    if _key in _type_map:
        try:
            globals()[_key] = _type_map[_key](_value)
        except ValueError:
            # ignore invalid casts; keep default
            pass


def generate_items_clustered(n_items: int,
                             num_clusters: int,
                             weight_min: int,
                             weight_max: int,
                             value_min: int,
                             value_max: int,
                             value_noise_range: int,
                             balanced: bool = True) -> List[Tuple[int, int, int]]:
    if num_clusters <= 0:
        raise ValueError("k (number of clusters) must be positive")
    if n_items <= 0:
        return []
    # Determine cluster sizes
    base_size = n_items // num_clusters
    remainder = n_items % num_clusters
    if balanced:
        cluster_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_clusters)]
    else:
        # Random positive partition of n_items across clusters
        # Ensure each cluster gets at least 1 when possible
        if num_clusters <= n_items:
            cuts = sorted(random.sample(range(1, n_items), num_clusters - 1))
            cuts = [0] + cuts + [n_items]
            cluster_sizes = [cuts[i+1] - cuts[i] for i in range(num_clusters)]
        else:
            # More clusters than items: some clusters will be empty
            cluster_sizes = [1] * n_items + [0] * (num_clusters - n_items)
            random.shuffle(cluster_sizes)

    items_local: List[Tuple[int, int, int]] = []
    item_idx = 0
    for c in range(num_clusters):
        size_c = cluster_sizes[c]
        if size_c == 0:
            continue
        weight_c = random.randint(weight_min, weight_max) if weight_max >= weight_min else weight_min
        base_value_c = random.randint(value_min, value_max) if value_max >= value_min else value_min
        for _ in range(size_c):
            noise = random.randint(-value_noise_range, value_noise_range) if value_noise_range > 0 else 0
            value = max(1, base_value_c + noise)
            weight = max(1, weight_c)
            items_local.append((item_idx, value, weight))
            item_idx += 1
    return items_local

num_clusters = int(globals().get("k", 10) or 10)
weight_min = int(globals().get("w_min", 10) or 10)
weight_max = int(globals().get("w_max", 1000) or 1000)
value_min = int(globals().get("v_min", 10) or 10)
value_max = int(globals().get("v_max", 10000) or 10000)
value_noise_range = int(globals().get("noise", 100) or 100)
balanced_clusters = int(globals().get("balanced_clusters", 1) or 1) == 1

# generate values and weights
items = generate_items_clustered(
        n_items=int(n),
        num_clusters=int(num_clusters),
        weight_min=int(weight_min),
        weight_max=int(weight_max),
        value_min=int(value_min),
        value_max=int(value_max),
        value_noise_range=int(value_noise_range),
        balanced=bool(balanced_clusters),
)

# compute total capacity as half of total item weight
total_weight = sum(w for (_, _, w) in items)
cap_total = max(1, total_weight // 2)

# partition capacity into m bins that sum to cap_total
if cap_total >= m:
    bins = random_partition(cap_total, m)
else:
    # Edge-case: fewer units of capacity than bins; assign 1 to first cap_total bins
    bins = [1] * cap_total + [0] * (m - cap_total)

# Helper function to map (i, j) to linear variable index
# Variable index for item i in bin j is: i * m + j + 1 (OPB uses 1-indexed)
def var_idx(i, j):
    return i * m + j + 1

# generate knapsack problem file
print(f"* Name: knapsack_clustered_n{n}_c{cap_total}_m{m}_k{num_clusters}_w[{weight_min},{weight_max}]_v[{value_min},{value_max}]_noise{value_noise_range}_balanced{balanced_clusters}")
print(f"* Variables: {n * m}")
print(f"* Constraints: {1}")
print("min: ", end = "")
# Objective function
for i in range(n):
    for j in range(m):
        print(" - %d x%d" % (items[i][1], var_idx(i, j)), end = "")
print(";")

# Capacity constraint: the total weight of the items must not exceed the capacity
c = 0
for j in range(m):
    for i in range(n):
        print(" +%d x%d" % (items[i][2], var_idx(i, j)), end="")
    print(" <= %d;" % (bins[j]))
    c = c + 1
    if c % 10 == 0:
        print()


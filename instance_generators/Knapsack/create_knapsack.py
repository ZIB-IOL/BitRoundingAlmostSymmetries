import subprocess
import sys

num_items = [100000, 200000, 500000, 750000, 1000000]
num_clusters = [50, 100, 200, 350, 500]
weights = [(50, 100000), (50, 500)]
values = (2**12, 2**22)
noises = [2**6, 2**12]
balanced_clusters = [0, 1]
clustered = 1

for n in num_items:
    for c in num_clusters:
        for w in weights:
            for s in noises:
                for b in balanced_clusters:
                    filename = f"knapsack_clustered_n{n}_m1_k{c}_w[{w[0]},{w[1]}]_v[{values[0]},{values[1]}]_noise{s}_balanced{b}.opb"
                    filepath = f"KnapsackOPB/{filename}"
                    with open(filepath, "w") as f_out:
                        subprocess.run([
                            "python", 
                            "generateKnapsack.py", 
                            f"n={n}", 
                            f"m=1", 
                            f"k={c}", 
                            f"w_min={w[0]}", 
                            f"w_max={w[1]}", 
                            f"v_min={values[0]}", 
                            f"v_max={values[1]}", 
                            f"noise={s}", 
                            f"balanced_clusters={b}"
                            ], stdout=f_out, check=True)
                sys.stdout.write(f"Created {filename}\n")
                sys.stdout.flush()
import subprocess
import random
import math
import sys

customers = [100, 500, 1000, 1500, 2000]
facilities = [20, 50, 100, 250, 500]
accuracies = [1, 3, 5, 8]

for c in customers:
    for f in facilities:
        for a in accuracies:
            capacity = int((1 + random.random()) * math.ceil(c / f))
            filename = f"cflp_{c}_{f}_{capacity}_{a}.lp"
            filepath = f"CFLP/{filename}"
            with open(filepath, "w") as f_out:
                subprocess.run([
                    "python",
                    "generateCFLP.py",
                    f"n={c}",
                    f"m={f}",
                    f"Cap={capacity}",
                    f"rounding_accuracy={a}",
                ], stdout=f_out, check=True)
            sys.stdout.write(f"Created {filename}\n")
            sys.stdout.flush()
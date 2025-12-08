import random
import math
import sys

# number of customers
n = 10

# number of facilities
m = 1

# fixed cost for facilities
C = 1e6

# Capacities of facilities
Cap = 10

# l-bit rounding
l = 0

# scaling of customer positions
scale = 2.0

# how many digits to round the coefficients to. Lower rounding accuracy leads to more symmetry. Range between 1 and 6.
rounding_accuracy = 5

# allow overriding defaults via CLI key=value arguments
_type_map = {
    "n": int,
    "m": int,
    "C": int,
    "Cap": int,
    "l": int,
    "scale": float,
    "rounding_accuracy": int,
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

# randomly generate positions of customers (scaled square arround 0)
X = []
Y = []
for j in range(n):
    x = scale * (random.random() - 0.5);
    y = scale * (random.random() - 0.5);
    X.append(x)
    Y.append(y)

# generate positions of facilities

# base for facilities
radius = 10.0
angle = 2 * math.pi / m

Fx = []
Fy = []
for i in range(m):
    x = radius * math.cos(angle * i)
    y = radius * math.sin(angle * i)
    Fx.append(x)
    Fy.append(y)

# compute range of distances
maxdist = 0.0
for i in range(m):
    for j in range(n):
        # compute distance
        x1 = X[j]
        x2 = Fx[i]
        y1 = Y[j]
        y2 = Fy[i]
        d = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

        if d > maxdist:
            maxdist = d

# compute scaling factor such that all distances are between 0 and 1
distscale = 1.0 / maxdist

# write LP file
print("minimize obj: ", end = "")
c = 0
for i in range(m):
    for j in range(n):

        # compute distance
        x1 = X[j]
        x2 = Fx[i]
        y1 = Y[j]
        y2 = Fy[i]
        d = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) * distscale;
        d = round(d, rounding_accuracy)
        d = (10 ** 6) * d
        d = int(round(d))


        # perform l-bit rounding. This is done before the solving process starts, not when creating the model.
        # d = int(Utils.transform_bits([int(d)], l)[0])

        print(" %+g x#%d#%d" % (d, i, j), end = "")

        c = c + 1
        if c % 10 == 0:
            print()

for i in range(m):
    print(" + %g y%d" % (C, i), end = "")
    
print()

# coupling conditions
print("Subject to")

c = 0
for i in range(m):
    print("C#%d#%d: " % (i,j), end="")
    for j in range(n):
        print("+ x#%d#%d" % (i, j), end="")

    c = c + 1
    if c % 10 == 0:
        print()

    print("- %g y%d <= 0" % (Cap,i))

# assignment constraints
for j in range(n):
    print("A#%d#%d: " % (i, j), end = "")
    for i in range(m):
        print(" + x#%d#%d" % (i, j), end = "")
    print(" == 1")

# variables
print("Binaries")
c = 0
for i in range(m):
    for j in range(n):
        print(" x#%d#%d" % (i, j), end="")
        c = c + 1
        if c % 10 == 0:
            print()
for i in range(m):
    print(" y%d" % (i), end="")
print()

print("End")

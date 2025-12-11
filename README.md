# Objective Coefficient Rounding and Almost Symmetries in Binary Programs
_Authors: Dominik Kuzinowicz, Paweł Lichocki, [Gioni Mexi](https://gionimexi.com), [Marc E. Pfetsch](https://www2.mathematik.tu-darmstadt.de/~pfetsch/index.en.html), [Sebastian Pokutta](https://www.pokutta.com), [Max Zimmer](https://maxzimmer.org)_

This repository contains experimental result data, instance generation scripts and instances from the paper `Objective Coefficient Rounding and Almost Symmetries in Binary Programs`. 

## Structure

### Results
The data for our results is divided per problem. Each problem class between `CLFP`, `Knapsack` and `PB` has a `Sym` and `No_Sym` version, indicating whether symmetry handling was enabled. The longtable PDFs contain the data collected in the `.csv` files in a combined table and each table is sorted by problem number as well as by seed, e.g. the first row for each problem is run with the same seed. The problem names corresponding to each number are stored in the respective `problems.py` file in the form of a dictionary.

### Instances and instance generators
Our testset are provided separately as `.opb.gz` files together with the scripts to generate the `CFLP` and `Knapsack` instances. As stated in the paper, the collection of `PB` instances are gathered from two sources, the [Pseudo-Boolean Competition 2024](https://www.cril.univ-artois.fr/PB24/details.html) and [translated MIPLIB 0-1 instances](https://zenodo.org/records/3870965).
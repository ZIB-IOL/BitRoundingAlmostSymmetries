# Objective Coefficient Rounding and Almost Symmetries in Binary Programs
_Authors: Dominik Kuzinowicz, [Paweł Lichocki](https://research.google/people/pawellichocki/?&type=google), [Gioni Mexi](https://gionimexi.com), [Marc E. Pfetsch](https://www2.mathematik.tu-darmstadt.de/~pfetsch/index.en.html), [Sebastian Pokutta](https://www.pokutta.com), [Max Zimmer](https://maxzimmer.org)_

This repository contains code to run experiments and experimental result data, instance generation scripts and instances from the paper ["Objective Coefficient Rounding and Almost Symmetries in Binary Programs"](https://arxiv.org/abs/2512.10507). 

## Structure and Usage

### Code
Experiments are started from [main.py](main.py) using the dictionary format of [Weights & Biases](https://wandb.ai/site/). It is structured as follows:

- `problem`: decides which problem to load from the dictionary in [problem_dict.py](problem_dict.py)
- `mode`: determines which solver to use and whether symmetry will be enabled
- `bit_num`: integer to decide $\ell$-bit rounding level
- `solving_time`: maximum solving time in seconds
- `seed`: seed for randomization in solvers

The two files [runner.py](runner.py) and [utils.py](utils.py) hold the code for the solving process and useful auxiliary functions, respectively.

### Results
The data for our results is divided per problem. Each problem class between `CLFP`, `Knapsack` and `PB` has a `Sym` and `No_Sym` version, indicating whether symmetry handling was enabled. The longtable PDFs contain the data collected in the `.csv` files in a combined table and each table is sorted by problem number as well as by seed, e.g. the first row for each problem is run with the same seed, the second rows for each problem are the same seed, etc. The problem names corresponding to each number are stored in the respective `problems.py` file in the form of a dictionary, as well as combined in one dictionary in [problem_dict.py](problem_dict.py).

### Instances and instance generators
Our testset are provided separately as `.opb.gz` files together with the scripts to generate the `CFLP` and `Knapsack` instances. As stated in the paper, the collection of `PB` instances are gathered from two sources, the [Pseudo-Boolean Competition 2024](https://www.cril.univ-artois.fr/PB24/details.html) and [translated MIPLIB 0-1 instances](https://zenodo.org/records/3870965).

## Citation
In case you find the paper or the implementation useful for your own research, please consider citing:
```
@misc{kuzinowicz2025objectivecoefficientroundingsymmetries,
      title={Objective Coefficient Rounding and Almost Symmetries in Binary Programs}, 
      author={Dominik Kuzinowicz and Paweł Lichocki and Gioni Mexi and Marc E. Pfetsch and Sebastian Pokutta and Max Zimmer},
      year={2025},
      eprint={2512.10507},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2512.10507}, 
}
```

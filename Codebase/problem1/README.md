# Problem 1

This folder contains the code for Problem 1.

## Objective

This program solves a 0/1 knapsack problem using three approaches:

- Classical brute-force search
- QUBO formulation solved by ExactSolver
- QUBO formulation solved by Simulated Annealing

The script also checks multiple penalty coefficients for the QUBO formulation and saves summary files for reporting.

## File

- `problem1_main.py`  
  Main script for Problem 1.

## How to run

Please use Python 3.

```bash
python3 problem1_main.py
```

## What the script does

The script is designed to:

- define the knapsack item values, weights, and capacity
- find the exact classical optimum by full enumeration
- build a QUBO using slack variables
- test multiple lambda values with `dimod.ExactSolver`
- run simulated annealing with multiple `num_reads` settings
- compare the best solutions from the three approaches
- save CSV and Markdown summary files for the report

## Default problem setup

The current code uses:

- seed: `10010022`
- capacity: `165`
- number of items: `10`
- slack bits: `8`
- ExactSolver lambda list: `0.5, 1.0, 2.0, 4.5, 6.0, 10.0`
- Simulated Annealing `num_reads`: `10, 100, 1000, 10000`

## Requirements

Please install the required Python packages before execution.  
Typical packages include:

- Python 3.x
- numpy
- pandas
- dimod
- dwave-neal

Depending on your environment, `neal` may be installed from the `dwave-neal` package.

## Output

The script saves several output files, including:

- `problem1_exact_results.csv`
- `problem1_sa_results.csv`
- `problem1_comparison_table.csv`
- `problem1_runtime_summary.csv`
- `problem1_summary.md`

## Notes

- This README is based on the current `problem1_main.py` contents.
- The script currently corresponds to a knapsack/QUBO workflow for Problem 1.
- Please refer to the submitted report for the final discussion and formatted results.

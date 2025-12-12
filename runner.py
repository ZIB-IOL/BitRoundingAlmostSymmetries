import os
import sys
import time
from typing import List, Optional, Tuple
import platform

import numpy as np
import wandb

from problem_dict import l_bit_dict
from utils import Utils, OPBParser

from pyscipopt import Model as SCIPModel, quicksum, SCIP_PARAMSETTING
from ortools.sat.python import cp_model

class Runner:

    def __init__(self, config, tmp_dir, debug):
        """
        Initialize runner. 
        """

        self.config = config
        self.tmp_dir = tmp_dir
        self.debug = debug

        self.is_htc = 'htc-' in platform.uname().node                                               # True if we are running on the HTC cluster
        self.is_coder = 'coder' in platform.uname().node and 'workspace' in platform.uname().node   # True if we are running on the coder workspace

        self.problem_instance_basedir = "./"        

        self.problem_name = l_bit_dict[int(self.config.problem)]
        
        assert os.path.exists(self.problem_name), f"Problem file {self.problem_name} not found."

        # Parameters
        self.L = self.config.bit_num or 0    # Number of bits in L-bit representation
        assert self.L >= 0 and type(self.L) == int, "L must be non-negative and an integer."
        self.epsilon = 2/(2**(self.L-1)-1) if self.L > 1 else 0 if self.L == 0 else float("inf")    # max distance from original objective value  
        self.timelimit = self.config.solving_time    

    def log_metrics(self):
        """Log metrics to wandb."""

        logging_dict = {
            "Objective value": self.objective_value,
            "Total time": self.total_solving_time,
        }
        wandb.log(logging_dict)

        summary_dict = {
            "Coefficients rounded to integer": self.has_been_rounded,
            "Total time limit reached": self.total_time_out_reached,
            "L-bit rounding": self.L if self.L > 0 else False,
            "Termination status": self.solving_status,
        }

        for key, value in summary_dict.items():
            wandb.run.summary[key] = value

    def init_model(self, x0: Optional[np.ndarray] = None, c: Optional[np.ndarray] = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Loads the MIPLIB model with given model name into SCIP model."""
        if verbose:
            sys.stdout.write(f"Loading MIPLIB model {self.problem_name}.\n")

        # Initializes model, feasible starting point x0 and integer coefficient vector c
        self.model = SCIPModel()

        self.read_problem()

        # save variables and objective terms for easier access when warm starting and changing objective
        self.vars = list(self.model.getVars())
        self.objective_terms = list(self.model.getObjective().terms.keys())
        self.objective_terms = [self.objective_terms[i][0] for i in range(len(self.objective_terms))]

        # Store rounded values of objective coefficients in array
        if c is None:
            self.original_coefficients = np.array(list(self.model.getObjective().terms.values()))
            c_rounded = np.round(self.original_coefficients)
            self.has_been_rounded = not np.array_equal(self.original_coefficients, c_rounded)
            self.original_coefficients = np.array([int(coeff) for coeff in c_rounded])

            if verbose:
                sys.stdout.write(f"Rounding objective coefficients to integer: {self.has_been_rounded}.\n")

    	    # L-bit rounding
            if self.L > 0:
                if verbose:
                    sys.stdout.write(f"Transforming objective coefficients to {self.L}-bit representation.\n")
                self.c = Utils.transform_bits(self.original_coefficients, self.L)
            else:
                self.c = self.original_coefficients

        # Set max solving time
        self.model.setParam("limits/time", self.timelimit)

        # Set seed for reproducibility
        if self.config.seed is not None:
            self.model.setParam("randomization/randomseedshift", self.config.seed)
        
        if verbose:
            sys.stdout.write("Model loaded successfully.\n")
        return x0, self.c
    
    def read_problem(self):
        """Reads in the problem from self.model, currently only works for SCIP and MIPLIB."""
        if "scip" in self.config.mode:
            problem_path = os.path.join(self.problem_instance_basedir, self.problem_name)
            assert os.path.exists(problem_path), f"Problem path {problem_path} does not exist."

            self.model.readProblem(problem_path)
            wandb.log({"Problem": f"{self.problem_name.split('/')[-1][:-4]}"})
    
    def scip_solving(self) -> Tuple[np.ndarray, float, float]:
        """Solves the model using SCIP."""
        t_start = time.time()
        x0, self.c = self.init_model(x0=None, c=None)

        self.model.freeReoptSolve()

        # Get dimensions
        m, n = len(self.vars), len(self.c)

        # Set objective and time limit
        self.model.chgReoptObjective(quicksum([(self.c[i] * (self.objective_terms[i])) for i in range(n)]))  
        self.model.setParam("limits/time", self.timelimit)

        # Solve model
        sys.stdout.write("Starting solving process...\n")
        sys.stdout.flush()
        self.model.optimize()
        solving_time = self.model.getSolvingTime()
        sys.stdout.write(f"Solving process complete. Elapsed time: {round(solving_time, 2)} seconds.\n")
        sys.stdout.flush()

        self.solving_status = self.model.getStatus()
        self.gap = self.model.getGap()
        wandb.log({"Primal-Dual Gap": self.gap})
        if self.solving_status in ["optimal", "timelimit"]:
            try:
                x0 = np.array([self.model.getVal(self.objective_terms[i]) for i in range(len(self.objective_terms))])
                obj_value = int(np.dot(self.original_coefficients, x0))
            except: 
                # Failsafe in case no feasible solution is found within time limit but the problem is not found to be infeasible
                x0 = None
                obj_value = float("inf")
        else:
            x0 = None
            obj_value = float("inf")
        t_elapsed = time.time() - t_start
        self.total_time_out_reached = self.model.getStatus() == "timelimit"

        wandb.log({"Solving nodes": self.model.getNTotalNodes()})

        if self.L > 0 and not self.use_relative_gap and self.solving_status != "infeasible":
            wandb.log({"L-bit rounded objective value": self.model.getObjVal()})

        
        presolve_time = self.model.getPresolvingTime()
        wandb.log({"Presolving time": presolve_time})
        wandb.log({"Solving time": solving_time})
        wandb.log({"Epsilon": self.epsilon})

        return x0, obj_value, t_elapsed

    def scip_no_symmetry_solving(self) -> Tuple[np.ndarray, float, float]:
        """Solves the model using SCIP."""
        t_start = time.time()
        x0, self.c = self.init_model(x0=None, c=None)

        self.model.freeReoptSolve()

        # Get dimensions
        m, n = len(self.vars), len(self.c)

        # Set objective and time limit
        self.model.chgReoptObjective(quicksum([(self.c[i] * (self.objective_terms[i])) for i in range(n)]))  
        self.model.setParam("limits/time", self.timelimit)
        self.model.setParam("misc/usesymmetry", 0)

        # Solve model
        sys.stdout.write("Starting solving process...\n")
        sys.stdout.flush()
        self.model.optimize()
        solving_time = self.model.getSolvingTime()
        sys.stdout.write(f"Solving process complete. Elapsed time: {round(solving_time, 2)} seconds.\n")
        sys.stdout.flush()

        self.solving_status = self.model.getStatus()
        self.gap = self.model.getGap()
        wandb.log({"Primal-Dual Gap": self.gap})
        if self.solving_status in ["optimal", "timelimit", "gaplimit"]:
            try:
                x0 = np.array([self.model.getVal(self.objective_terms[i]) for i in range(len(self.objective_terms))])
                obj_value = int(np.dot(self.original_coefficients, x0))
            except: 
                # Failsafe in case no feasible solution is found within time limit but the problem is not found to be infeasible
                x0 = None
                obj_value = float("inf")
        else:
            x0 = None
            obj_value = float("inf")
        t_elapsed = time.time() - t_start
        self.total_time_out_reached = self.model.getStatus() == "timelimit"

        wandb.log({"Solving nodes": self.model.getNTotalNodes()})

        if self.L > 0 and self.solving_status != "infeasible":
            wandb.log({"L-bit rounded objective value": self.model.getObjVal()})
        
        presolve_time = self.model.getPresolvingTime()
        wandb.log({"Presolving time": presolve_time})
        wandb.log({"Solving time": solving_time})
        wandb.log({"Epsilon": self.epsilon})

        return x0, obj_value, t_elapsed


    def ortools_cp_solving(self):
        """
        Direct OR-Tools CP-SAT solving without bit scaling (baseline comparison).
        Works directly with OPB files. Supports optional L-bit rounding.
        """
        t_start = time.time()
        self.no_symmetry = False
        
        # Parse the OPB file
        parser = OPBParser(self.problem_name)
        parser.parse()
        
        sys.stdout.write(f"Parsed OPB file: {parser.num_vars} variables, {parser.num_constraints} constraints\n")
        
        # Log problem name to wandb
        wandb.log({"Problem": f"{self.problem_name.split('/')[-1][:-4]}"})
        
        # Get original objective coefficients
        self.original_coefficients = np.array(parser.objective_coeffs)
        self.coefficients = self.original_coefficients.copy()
        
        # Apply L-bit transformation if specified
        if self.L > 0:
            sys.stdout.write(f"Applying {self.L}-bit rounding to objective coefficients\n")
            sys.stdout.flush()
            self.coefficients = Utils.transform_bits(self.original_coefficients, self.L)
        
        self.model, self.phase_variables = parser.build_model()
        
        # Solve with coefficients (original or L-bit transformed) using full time limit
        result = self._solve_ortools_phase_optimized(
            parser, self.coefficients, None, self.timelimit
        )
        
        if result['status'] not in ['OPTIMAL', 'FEASIBLE']:
            self.objective_value = float("inf")
            self.solving_status = "INFEASIBLE" if result['status'] == 'INFEASIBLE' else "TIMEOUT"
            self.total_solving_time = time.time() - t_start
            self.total_time_out_reached = result['status'] == 'UNKNOWN'
            self.has_been_rounded = False
            return None, self.objective_value, self.total_solving_time
        
        current_solution = result['solution']
        
        # Calculate final results using original coefficients
        t_elapsed = time.time() - t_start
        self.objective_value = self._calculate_objective_value(parser, self.original_coefficients, current_solution)
        self.total_time_out_reached = t_elapsed >= self.timelimit
        self.total_solving_time = t_elapsed
        self.has_been_rounded = False
        
        if not self.total_time_out_reached:
            self.solving_status = "OPTIMAL" if result['status'] == 'OPTIMAL' else "FEASIBLE"
        else:
            self.solving_status = "TIMEOUT"
        
        # Log results to wandb
        wandb.log({"Total time": self.total_solving_time})
        wandb.log({"Termination status": self.solving_status})
        wandb.log({"Epsilon": self.epsilon})
        wandb.log({"Solving time": self.solving_time})
        if self.L > 0:
            l_bit_objective = self._calculate_objective_value(parser, self.coefficients, current_solution)
            wandb.log({"L-bit rounded objective value": l_bit_objective})
        
        return current_solution, self.objective_value, t_elapsed

    def cp_no_symmetry_solving(self):
        """
        Direct OR-Tools CP-SAT solving without bit scaling (baseline comparison).
        Works directly with OPB files. Supports optional L-bit rounding.
        """
        t_start = time.time()
        self.no_symmetry = True
        
        # Parse the OPB file
        parser = OPBParser(self.problem_name)
        parser.parse()
        
        sys.stdout.write(f"Parsed OPB file: {parser.num_vars} variables, {parser.num_constraints} constraints\n")
        
        # Log problem name to wandb
        wandb.log({"Problem": f"{self.problem_name.split('/')[-1][:-4]}"})
        
        # Get original objective coefficients
        self.original_coefficients = np.array(parser.objective_coeffs)
        self.coefficients = self.original_coefficients.copy()
        
        # Apply L-bit transformation if specified
        if self.L > 0:
            sys.stdout.write(f"Applying {self.L}-bit rounding to objective coefficients\n")
            sys.stdout.flush()
            self.coefficients = Utils.transform_bits(self.original_coefficients, self.L)
            wandb.log({"L-bit rounding applied": True})
        
        self.model, self.phase_variables = parser.build_model()
        
        # Solve with coefficients (original or L-bit transformed) using full time limit
        result = self._solve_ortools_phase_optimized(
            parser, self.coefficients, None, self.timelimit
        )
        
        if result['status'] not in ['OPTIMAL', 'FEASIBLE']:
            self.objective_value = float("inf")
            self.solving_status = "INFEASIBLE" if result['status'] == 'INFEASIBLE' else "TIMEOUT"
            self.total_solving_time = time.time() - t_start
            self.total_time_out_reached = result['status'] == 'UNKNOWN'
            self.has_been_rounded = False
            return None, self.objective_value, self.total_solving_time
        
        current_solution = result['solution']
        
        # Calculate final results using original coefficients
        t_elapsed = time.time() - t_start
        self.objective_value = self._calculate_objective_value(parser, self.original_coefficients, current_solution)
        self.total_time_out_reached = t_elapsed >= self.timelimit
        self.total_solving_time = t_elapsed
        self.has_been_rounded = False
        
        if not self.total_time_out_reached:
            self.solving_status = "OPTIMAL" if result['status'] == 'OPTIMAL' else "FEASIBLE"
        else:
            self.solving_status = "TIMEOUT"
        
        # Log results to wandb
        wandb.log({"Total time": self.total_solving_time})
        wandb.log({"Termination status": self.solving_status})
        wandb.log({"Epsilon": self.epsilon})
        wandb.log({"Solving time": self.solving_time})
        if self.L > 0:
            l_bit_objective = self._calculate_objective_value(parser, self.coefficients, current_solution)
            wandb.log({"L-bit rounded objective value": l_bit_objective})
        
        return current_solution, self.objective_value, t_elapsed

    
    def _solve_ortools_phase_optimized(self, parser: OPBParser, objective_coeffs: np.ndarray, warm_start_solution: Optional[np.ndarray], time_limit: float):
        """
        Optimized solve with enhanced infeasibility handling and hint recovery.
        
        Args:
            parser: Parsed OPB file
            objective_coeffs: Coefficients for the objective function in this phase
            warm_start_solution: Previous solution to use as warm start (None for first phase)
            time_limit: Time limit for this phase in seconds
            
        Returns:
            Dictionary with 'status', 'solution', and 'objective_value'
        """
        # Set objective function using the parsed objective variable indices
        if len(objective_coeffs) > 0 and len(parser.objective_vars) > 0:
            obj_expr = []
            for coeff, var_idx in zip(objective_coeffs, parser.objective_vars):
                if var_idx < len(self.phase_variables) and coeff != 0:
                    obj_expr.append(int(coeff) * self.phase_variables[var_idx])
            
            if obj_expr:
                if parser.is_minimization:
                    self.model.Minimize(sum(obj_expr))
                else:
                    self.model.Maximize(sum(obj_expr))
        
        # Create solver with enhanced parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = True
        if self.no_symmetry:
            solver.parameters.symmetry_level = 0
        else:
            solver.parameters.symmetry_detection_deterministic_time_limit = 0.01 * time_limit    # higher time limit (default 1 second) to allow deeper symmetry detection
        solver.parameters.num_workers = 8
        solver.parameters.cp_model_use_sat_presolve = False

        # Set random seed for reproducibility (similar to SCIP)
        if self.config.seed is not None:
            solver.parameters.random_seed = self.config.seed
        
        status = solver.Solve(self.model)
        
        # Extract results
        result = {
            'status': solver.StatusName(status),
            'solution': None,
            'objective_value': None
        }
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = []
            for var in self.phase_variables:
                solution.append(solver.Value(var))
            result['solution'] = np.array(solution)
            
            if len(objective_coeffs) > 0:
                result['objective_value'] = solver.ObjectiveValue()
        
        self.solving_time = solver.user_time
        
        return result
    
    def _calculate_objective_value(self, parser: OPBParser, coefficients: np.ndarray, solution: np.ndarray) -> int:
        """Calculate the objective value using the parsed variable mapping."""
        obj_value = 0
        if solution is None or len(solution) == 0:
            return float("inf")
        for coeff, var_idx in zip(coefficients, parser.objective_vars):
            if var_idx < len(solution):
                obj_value += coeff * solution[var_idx]
        return int(obj_value)



    def run(self):
        if self.config.mode == "scip":
            self.result, self.objective_value, self.total_solving_time = self.scip_solving()
        elif self.config.mode == "ortools_cp":
            self.result, self.objective_value, self.total_solving_time = self.ortools_cp_solving()
        elif self.config.mode == "scip_no_symmetry":
            self.result, self.objective_value, self.total_solving_time = self.scip_no_symmetry_solving()
        elif self.config.mode == "cp_no_symmetry":
            self.result, self.objective_value, self.total_solving_time = self.cp_no_symmetry_solving()
            
        self.log_metrics()


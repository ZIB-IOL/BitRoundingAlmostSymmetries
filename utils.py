import numpy as np
import re
from ortools.sat.python import cp_model
import time
import sys
from pathlib import Path
from typing import List, Tuple, Union

class Utils:
    """Utility functions."""
    @staticmethod
    def fill_dict_with_none(d):
        """Recursively fill a dictionary with None values."""
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                new_dict[k] = Utils.fill_dict_with_none(v)
            else:
                new_dict[k] = None
        return new_dict
    
    @staticmethod
    def update_config_with_default(config, defaults):
        """Update config with defaults for None values."""
        for k, v in defaults.items():
            if config.get(k) is None:
                config[k] = v
        return config
    
    @staticmethod
    def transform_bits(c: np.ndarray, L: int) -> np.ndarray:
        """
        Transform integer vector into L-bit representation.
        For each number, keep highest 1 bit and L-1 bits to its right, set rest to 0.
        
        Args:
            c: Input integer vector
            L: Number of bits to keep (including highest 1 bit)
        
        Returns:
            Transformed integer vector
        """
        if L <= 0:
            return c
        # Convert to absolute values and keep signs for later
        signs = np.sign(c)
        c_abs = np.abs(c)
        
        result = []
        for num in c_abs:
            if num == 0:
                result.append(0)
                continue
                
            # Convert to binary string without '0b' prefix
            bin_str = bin(int(num))[2:]

            # Keep num if binary string is at most L bits
            if len(bin_str) <= L:
                result.append(num)
                continue
            
            # Keep L bits starting from highest 1, set rest to 0
            significant_bits = bin_str[:L]  # Keep first L bits
            zeros_to_add = len(bin_str) - L  # Number of zeros to append
            if zeros_to_add > 0:
                new_bin = significant_bits + '0' * zeros_to_add
            else:
                new_bin = bin_str
                
            result.append(int(new_bin, 2))
        
        # Convert back to numpy array and restore signs
        return signs * np.array(result)


    @staticmethod
    def parse_minimize_expression(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the minimize expression
        minimize_match = re.search(r'solve minimize (.*?);', content)
        if not minimize_match:
            return None
        
        minimize_expr = minimize_match.group(1)
        
        # Split the expression into terms, handling both + and - signs
        # First, add spaces around operators for consistent splitting
        minimize_expr = minimize_expr.replace(' + ', ' + ').replace(' - ', ' - ')
        # Split on spaces, which will give us terms with their signs
        terms = minimize_expr.split()
        
        # Parse each term
        coefficients = []
        variables = []
        
        for i, term in enumerate(terms):
            if term in ['+', '-']:
                continue
                
            # Handle terms with coefficients (e.g., "2*x359" or "-2*x359")
            if '*' in term:
                coef, var = term.split('*')
                # Check if previous term was a minus sign
                if i > 0 and terms[i-1] == '-':
                    coefficients.append(-int(coef))  # Make coefficient negative
                else:
                    coefficients.append(int(coef))  # Convert to int
                variables.append(var)
            # Handle terms without coefficients (e.g., "x235" or "-x235")
            else:
                if term.startswith('-'):
                    coefficients.append(-1)  # Already an int
                    variables.append(term[1:])  # Remove the minus sign
                else:
                    # Check if previous term was a minus sign
                    if i > 0 and terms[i-1] == '-':
                        coefficients.append(-1)  # Make coefficient negative
                    else:
                        coefficients.append(1)  # Already an int
                    variables.append(term)
        
        return coefficients, variables
    
    @staticmethod
    def read_model_without_solve(file_path):
        """
        Reads a MiniZinc model file and returns:
        1. A list of all lines except the solve statement
        2. The coefficients and variables from the objective function
        
        Returns:
            tuple: (model_lines, coefficients, variables)
                - model_lines: list of strings containing the model without solve statement
                - coefficients: list of floats from the objective function
                - variables: list of strings from the objective function
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the solve statement line
        solve_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('solve'):
                solve_line_index = i
                break
        
        if solve_line_index == -1:
            raise ValueError("No solve statement found in the model")
        
        # Get all lines except the solve statement
        model_lines = [line for i, line in enumerate(lines) if i != solve_line_index]
        
        # Get the objective function components
        coefficients, variables = Utils.parse_minimize_expression(file_path)
        
        return model_lines, coefficients, variables
    
    @staticmethod
    def get_current_objective_function(coefficients, variable_names, mu):
        """
        Returns the current objective function as a string for MiniZinc model.
        """
        c_mu = np.sign(coefficients) * np.floor(np.abs(coefficients) / mu)
        objective_string = ""
        for i in range(len(c_mu)):
            objective_string += f" {int(c_mu[i])}*{variable_names[i]} "
            objective_string += "+"
        objective_string = objective_string[:-1]
        objective_string += ";"
        return objective_string
    
    def determine_bit_counts(coeffs):
        abs_coeffs = np.abs(coeffs)
        bit_counts = []
        highest_bit_positions = []
        for coeff in abs_coeffs:
            bit_count, highest_bit_position = Utils.count_nonzero_bits(coeff)
            bit_counts.append(bit_count)
            highest_bit_positions.append(highest_bit_position)

        return max(bit_counts), max(highest_bit_positions)
    
    def count_nonzero_bits(integer):
        """
        Count the number of nonzero bits (1s) in the binary representation of an integer.
        
        Args:
            integer (int): The integer to count bits for
            
        Returns:
            tuple: (nonzero_bit_count, highest_bit_position)
                   where highest_bit_position counts from right to left, starting at 1
            
        Examples:
            >>> count_nonzero_bits(5)  # 5 = 101 in binary
            (2, 3)  # 2 ones, highest bit at position 3
            >>> count_nonzero_bits(10)  # 10 = 1010 in binary
            (2, 4)  # 2 ones, highest bit at position 4
            >>> count_nonzero_bits(0)   # 0 = 0 in binary
            (0, 0)  # 0 ones, no highest bit
        """
        # Handle negative numbers by converting to positive
        if integer < 0:
            integer = abs(integer)
        
        # Convert to binary string and count the '1' characters
        binary_string = bin(integer)
        # bin() returns a string like '0b101', so we count '1's after the '0b' prefix
        count = binary_string.count('1')
        
        # Also return the highest bit position (length minus 2 for '0b' prefix)
        highest_bit_position = len(binary_string) - 2
        
        return count, highest_bit_position
    

class OPBParser:
    """Optimized parser for OPB files that focuses on objective function parsing."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.objective_coeffs = []
        self.objective_vars = []
        self.is_minimization = True
        self.num_vars = 0
        self.num_constraints = 0
        
    def parse(self):
        """Parse the OPB file - extract objective and count constraints."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        constraint_count = 0
        
        # Parse file
        for line in lines:
            line = line.strip()
            if line.startswith('* #variable='):
                self.num_vars = int(line.split('#variable=')[1].split()[0])
            elif line.startswith('*   Variables        :'):
                self.num_vars = int(line.split('*   Variables        :')[1].split()[0])
            elif line.startswith('* Variables:'):
                self.num_vars = int(line.split('* Variables:')[1].split()[0])
            elif line.startswith('* Constraints:'):
                self.num_constraints = int(line.split('* Constraints:')[1].split()[0])
            elif line.startswith('* #constraint='):
                self.num_constraints = int(line.split('#constraint=')[1].split()[0])
            elif line.startswith('*   Constraints      :'):
                self.num_constraints = int(line.split('*   Constraints      :')[1].split()[0])
            elif line.startswith('* Constraints:'):
                self.num_constraints = int(line.split('* Constraints:')[1].split()[0])
            elif line.startswith('min:') or line.startswith('max:'):
                self._parse_objective(line)
            elif line and not line.startswith('*') and not line.startswith('min:') and not line.startswith('max:'):
                if '>=' in line or '<=' in line or '=' in line:
                    constraint_count += 1
                    
        # Store actual constraint count 
        if self.num_constraints == 0:
            self.num_constraints = constraint_count
    
    def _parse_objective(self, line: str):
        """Parse the objective function line."""
        self.is_minimization = line.startswith('min:')
        # Remove 'min:' or 'max:' and the trailing semicolon
        obj_line = line[4:].rstrip(';').strip()
        
        # Parse coefficients and variables using regex
        pattern = r'([+-]?\s*\d+)\s+([x]\d+)'
        matches = re.findall(pattern, obj_line)
        
        for coeff_str, var_str in matches:
            coeff = int(coeff_str.replace(' ', ''))
            var_index = int(var_str[1:]) - 1  # Convert x1 to index 0
            
            self.objective_coeffs.append(coeff)
            self.objective_vars.append(var_index)
    
    def get_max_var_index(self):
        """Get the maximum variable index used."""
        if not self.objective_vars:
            return self.num_vars
        return max(max(self.objective_vars) + 1, self.num_vars)


    
    def _add_constraint_to_model(self, model, variables, line: str):
        """Add a single constraint to the CP model."""
        # Split by comparison operator
        if '>=' in line:
            parts = line.split('>=')
            operator = '>='
        elif '<=' in line:
            parts = line.split('<=')
            operator = '<='
        elif '=' in line:
            parts = line.split('=')
            operator = '='
        else:
            return
        
        lhs = parts[0].strip()
        rhs = int(parts[1].strip().rstrip(';'))
        
        # Parse left-hand side
        pattern = r'([+-]?\s*\d+)\s+([x]\d+)'
        matches = re.findall(pattern, lhs)
        
        # Build linear expression
        expr = []
        for coeff_str, var_str in matches:
            coeff = int(coeff_str.replace(' ', ''))
            var_index = int(var_str[1:]) - 1  # Convert x1 to index 0
            
            if var_index < len(variables):
                expr.append(coeff * variables[var_index])
        
        if expr:  # Only add constraint if we have variables
            if operator == '>=':
                model.Add(sum(expr) >= rhs)
            elif operator == '<=':
                model.Add(sum(expr) <= rhs)
            elif operator == '=':
                model.Add(sum(expr) == rhs)

    def build_model(self):
        """Build a CP model from the OPB file."""
        st = time.time()
        model = cp_model.CpModel()
        # Create variables
        max_var_idx = self.get_max_var_index()
        phase_variables = []
        for i in range(max_var_idx):
            phase_variables.append(model.NewBoolVar(f'x{i+1}'))
        
        # Add constraints using the efficient parser method
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line and not line.startswith('*') and not line.startswith('min:') and not line.startswith('max:'):
                if '>=' in line or '<=' in line or '=' in line:
                    self._add_constraint_to_model(model, phase_variables, line)
        sys.stdout.write(f"Model building time: {time.time() - st:.2f} seconds\n")

        return model, phase_variables

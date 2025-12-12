import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import wandb

from runner import Runner
from utils import Utils

debug = '--debug' in sys.argv
defaults = dict(
    seed=0,                                         # Seed for randomness
    problem=0,                                      # problem file - using integer mapping from opb_problem_dict
    mode="scip",                        # "scip", "scip"
    bit_num=0,                                      # number of bits to consider. 0 means all. 
    solving_time=60,                                # max solving time in seconds - reduced for testing
) 

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)

@contextmanager
def tempdir():
    path = tempfile.mkdtemp()   # Creates a temporary directory under /tmp, which is automatically cleanup by the script (and by SLURM when on the HTC Cluster)
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))

with tempdir() as tmp_dir:
    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)

    # If exists, delete the core dump file from the directory of main.py
    if os.path.exists('core'):
        os.remove('core')

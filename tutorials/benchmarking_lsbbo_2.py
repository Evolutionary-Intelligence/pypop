"""Before running this script, please first run the following script to generate necessary data:
https://github.com/Evolutionary-Intelligence/pypop/blob/main/tutorials/benchmarking_lsbbo_1.py
"""

import os
import time
import pickle
import argparse
import importlib
import json
import yaml
from typing import Dict, Type, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np

import pypop7.benchmarks.continuous_functions as cf


@dataclass
class ExperimentConfig:
    """Centralized experiment configuration"""
    max_function_evaluations_multiplier: int = 100000
    max_runtime_hours: float = 3.0
    fitness_threshold: float = 1e-10
    saving_fitness: int = 2000
    boundary_range: float = 10.0
    sigma_value: float = 20.0 / 3.0
    random_seed: int = 2022
    verbose_level: int = 0
    results_folder: str = "pypop7_benchmarks_lso"


@dataclass
class OptimizerConfig:
    module_path: str
    class_name: str
    requires_sigma: bool = False


# Centralized optimizer configurations
OPTIMIZER_CONFIGS: Dict[str, OptimizerConfig] = {
    "PRS": OptimizerConfig("pypop7.optimizers.rs.prs", "PRS", True),
    "SRS": OptimizerConfig("pypop7.optimizers.rs.srs", "SRS", True),
    "GS": OptimizerConfig("pypop7.optimizers.rs.gs", "GS", True),
    "BES": OptimizerConfig("pypop7.optimizers.rs.bes", "BES", True),
    "HJ": OptimizerConfig("pypop7.optimizers.ds.hj", "HJ", True),
    "NM": OptimizerConfig("pypop7.optimizers.ds.nm", "NM", True),
    "POWELL": OptimizerConfig("pypop7.optimizers.ds.powell", "POWELL", True),
    "FEP": OptimizerConfig("pypop7.optimizers.ep.fep", "FEP", True),
    "GENITOR": OptimizerConfig("pypop7.optimizers.ga.genitor", "GENITOR", True),
    "G3PCX": OptimizerConfig("pypop7.optimizers.ga.g3pcx", "G3PCX", True),
    "GL25": OptimizerConfig("pypop7.optimizers.ga.gl25", "GL25", True),
    "COCMA": OptimizerConfig("pypop7.optimizers.cc.cocma", "COCMA", True),
    "HCC": OptimizerConfig("pypop7.optimizers.cc.hcc", "HCC", True),
    "SPSO": OptimizerConfig("pypop7.optimizers.pso.spso", "SPSO", True),
    "SPSOL": OptimizerConfig("pypop7.optimizers.pso.spsol", "SPSOL", True),
    "CLPSO": OptimizerConfig("pypop7.optimizers.pso.clpso", "CLPSO", True),
    "CCPSO2": OptimizerConfig("pypop7.optimizers.pso.ccpso2", "CCPSO2", True),
    "CDE": OptimizerConfig("pypop7.optimizers.de.cde", "CDE"),
    "JADE": OptimizerConfig("pypop7.optimizers.de.jade", "JADE"),
    "SHADE": OptimizerConfig("pypop7.optimizers.de.shade", "SHADE"),
    "SCEM": OptimizerConfig("pypop7.optimizers.cem.scem", "SCEM"),
    "MRAS": OptimizerConfig("pypop7.optimizers.cem.mras", "MRAS"),
    "DSCEM": OptimizerConfig("pypop7.optimizers.cem.dscem", "DSCEM"),
    "UMDA": OptimizerConfig("pypop7.optimizers.eda.umda", "UMDA", True),
    "EMNA": OptimizerConfig("pypop7.optimizers.eda.emna", "EMNA", True),
    "RPEDA": OptimizerConfig("pypop7.optimizers.eda.rpeda", "RPEDA", True),
    "XNES": OptimizerConfig("pypop7.optimizers.nes.xnes", "XNES", True),
    "SNES": OptimizerConfig("pypop7.optimizers.nes.snes", "SNES", True),
    "R1NES": OptimizerConfig("pypop7.optimizers.nes.r1nes", "R1NES", True),
    "VDCMA": OptimizerConfig("pypop7.optimizers.nes.vdcma", "VDCMA", True),
    "CMAES": OptimizerConfig("pypop7.optimizers.es.cmaes", "CMAES", True),
    "FMAES": OptimizerConfig("pypop7.optimizers.es.fmaes", "FMAES", True),
    "RMES": OptimizerConfig("pypop7.optimizers.es.rmes", "RMES", True),
    "LMMAES": OptimizerConfig("pypop7.optimizers.es.lmmaes", "LMMAES", True),
    "MMES": OptimizerConfig("pypop7.optimizers.es.mmes", "MMES", True),
    "LMCMA": OptimizerConfig("pypop7.optimizers.es.lmcma", "LMCMA", True),
    "LAMCTS": OptimizerConfig("pypop7.optimizers.bo.lamcts", "LAMCTS", True),
    }


def load_config(config_file: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from file or use defaults"""
    config = ExperimentConfig()
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    config_data = json.load(f)
                elif config_file.endswith(('.yml', '.yaml')):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError("Config file must be JSON or YAML format")
            
            # Update config with loaded values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            print(f"Configuration loaded from {config_file}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")
            print("Using default configuration")

    return config


def save_config_template(filename: str = "config_template.yaml") -> None:
    config = ExperimentConfig()
    config_dict = asdict(config)

    simple_config = {k: v for k, v in config_dict.items()}

    try:
        with open(filename, 'w') as f:
            yaml.dump(simple_config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration template saved to {filename}")
    except ImportError:
        json_filename = filename.replace('.yaml', '.json').replace('.yml', '.json')
        with open(json_filename, 'w') as f:
            json.dump(simple_config, f, indent=2)
        print(f"Configuration template saved to {json_filename} (YAML not available)")


def get_optimizer_class(optimizer_name: str) -> Type[Any]:
    if optimizer_name not in OPTIMIZER_CONFIGS:
        available_optimizers = ", ".join(sorted(OPTIMIZER_CONFIGS.keys()))
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available optimizers: {available_optimizers}"
        )

    config = OPTIMIZER_CONFIGS[optimizer_name]
    try:
        module = importlib.import_module(config.module_path)
        return getattr(module, config.class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import {config.class_name} from {config.module_path}: {e}"
        )


def requires_sigma(optimizer_name: str) -> bool:
    return OPTIMIZER_CONFIGS.get(optimizer_name, OptimizerConfig("", "")).requires_sigma


class Experiment(object):
    def __init__(self, index: int, function: Any, seed: int, ndim_problem: int, config: ExperimentConfig):
        self.index, self.seed = index, seed
        self.function, self.ndim_problem = function, ndim_problem
        self.config = config
        self._folder = config.results_folder
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        self._file = os.path.join(self._folder, "Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle")

    def run(self, optimizer_class: Type[Any]) -> None:
        problem = {
            "fitness_function": self.function,
            "ndim_problem": self.ndim_problem,
            "upper_boundary": self.config.boundary_range * np.ones((self.ndim_problem,)),
            "lower_boundary": -self.config.boundary_range * np.ones((self.ndim_problem,)),
        }

        options = {
            "max_function_evaluations": self.config.max_function_evaluations_multiplier * self.ndim_problem,
            "max_runtime": int(self.config.max_runtime_hours * 3600),  # Convert to seconds
            "fitness_threshold": self.config.fitness_threshold,
            "seed_rng": self.seed,
            "saving_fitness": self.config.saving_fitness,
            "verbose": self.config.verbose_level,
        }

        # Add sigma parameter if required
        if requires_sigma(optimizer_class.__name__):
            options["sigma"] = self.config.sigma_value

        solver = optimizer_class(problem, options)
        results = solver.optimize()

        file = self._file.format(
            solver.__class__.__name__,
            solver.fitness_function.__name__,
            solver.ndim_problem,
            self.index,
        )

        with open(file, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    def __init__(self, start: int, end: int, ndim_problem: int, config: ExperimentConfig):
        self.start, self.end = start, end
        self.ndim_problem = ndim_problem
        self.config = config

        # Test functions for local search ability
        self.functions = [
            cf.sphere,
            cf.cigar,
            cf.discus,
            cf.cigar_discus,
            cf.ellipsoid,
            cf.different_powers,
            cf.schwefel221,
            cf.step,
            cf.rosenbrock,
            cf.schwefel12,
        ]

        self.seeds = np.random.default_rng(config.random_seed).integers(
            np.iinfo(np.int64).max, size=(len(self.functions), 50)
        )

    def run(self, optimizer_class: Type[Any]) -> None:
        for index in range(self.start, self.end + 1):
            print(f"* experiment: {index} ***:")
            for i, f in enumerate(self.functions):
                start_time = time.time()
                print(f"  * function: {f.__name__}:")
                try:
                    experiment = Experiment(
                        index, f, self.seeds[i, index], self.ndim_problem, self.config
                    )
                    experiment.run(optimizer_class)
                    print(f"    runtime: {time.time() - start_time:.5e}.")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    print(f"    runtime: {time.time() - start_time:.5e}.")


def validate_arguments(args: argparse.Namespace) -> None:
    if not (0 <= args.start < 50):
        raise ValueError("start must be between 0 and 49")
    if not (0 <= args.end < 50):
        raise ValueError("end must be between 0 and 49")
    if args.start > args.end:
        raise ValueError("start must be <= end")
    if args.ndim_problem <= 0:
        raise ValueError("ndim_problem must be positive")
    if args.optimizer not in OPTIMIZER_CONFIGS:
        available = ", ".join(sorted(OPTIMIZER_CONFIGS.keys()))
        raise ValueError(f"Unknown optimizer: {args.optimizer}. Available: {available}")


def main() -> None:
    start_runtime = time.time()

    parser = argparse.ArgumentParser(
        description="Run PyPop7 benchmarking experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        required=True,
        help="Starting index of experiments (0-49)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        required=True,
        help="Ending index of experiments (0-49)",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        required=True,
        choices=list(OPTIMIZER_CONFIGS.keys()),
        help="Optimizer to use",
    )
    parser.add_argument(
        "--ndim_problem",
        "-d",
        type=int,
        default=2000,
        help="Dimension of fitness function",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file (JSON or YAML format)",
    )
    parser.add_argument(
        "--save-config-template",
        action="store_true",
        help="Save configuration template and exit",
    )

    args = parser.parse_args()

    try:
        if args.save_config_template:
            save_config_template()
            return 0

        config = load_config(args.config)

        validate_arguments(args)
        optimizer_class = get_optimizer_class(args.optimizer)

        print(f"Starting experiments with {args.optimizer} optimizer")
        print(f"Experiments: {args.start} to {args.end}")
        print(f"Problem dimension: {args.ndim_problem}")
        print(f"Configuration: {config}")

        experiments = Experiments(args.start, args.end, args.ndim_problem, config)
        experiments.run(optimizer_class)

        print(f"Total runtime: {time.time() - start_runtime:.5e}.")

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
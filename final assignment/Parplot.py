import os
import sys

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.api as sm

from platypus.algorithms import EpsMOEA, NSGAIII, MOEAD
from ema_workbench.analysis import prim, cart, dimensional_stacking, parcoords
from ema_workbench import ema_logging, load_results, MultiprocessingEvaluator, Constraint, Scenario
from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from ema_workbench.em_framework.optimization import ArchiveLogger, EpsilonProgress, epsilon_nondominated, Convergence, rebuild_platypus_population

from ema_workbench import (
    HypervolumeMetric,
    GenerationalDistanceMetric,
    EpsilonIndicatorMetric,
    InvertedGenerationalDistanceMetric,
    SpacingMetric,
)

from ema_workbench.em_framework.optimization import to_problem
from SALib.analyze import sobol
from problem_formulation import get_model_for_problem_formulation


if __name__ == '__main__':

    problem_formulation_id = 8
    dike_model, planning_step = get_model_for_problem_formulation(problem_formulation_id)

    for scenario_num in range(1, 6):
        print(scenario_num)

        merge_archs = []
        for i in range(5):
            arch = ArchiveLogger.load_archives(os.path.join('mordm', f'results_optimize_box_1_max_{i}.tar.gz'))
            for key in arch.keys():
                arch[key] = arch[key].drop(columns='Unnamed: 0')
            merge_archs.extend([val for val in arch.values()])

        buf = pd.concat(merge_archs)

        limits = parcoords.get_limits(buf[[o.name for o in dike_model.outcomes]])
        axes = parcoords.ParallelAxes(limits)
        axes.plot(buf[[o.name for o in dike_model.outcomes]])
        fig = plt.gcf()
        fig.savefig(os.path.join(
            'mordm', f'Parallel_plot_optimize_box_{scenario_num}.png'))
        plt.show()
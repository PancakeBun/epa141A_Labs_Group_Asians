
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


def dummy_model(problem_id):
    model, planning_step = get_model_for_problem_formulation(problem_id)
    for u in model.levers:
        u.name = 'z_' + '_'.join(u.name.split(" "))
        u.name = u.name.replace('.', 'z')
    for u in model.outcomes:
        u.name = 'z_' + '_'.join(u.name.split(" "))
        u.name = u.name.replace('.', 'z')
    for u in model.uncertainties:
        u.name = 'z_' + '_'.join(u.name.split(" "))
        u.name = u.name.replace('.', 'z')
        return model


def calculate_metrics(archives, reference_set, problem_id):
    model = dummy_model(problem_id)
    problem = to_problem(model, searchover="levers")
    # hv = HypervolumeMetric(reference_set, problem)
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    ig = InvertedGenerationalDistanceMetric(reference_set, problem, d=1)
    sm = SpacingMetric(problem)
    metrics = []
    for nfe, archive in archives.items():
        scores = {
            "generational_distance": gd.calculate(archive),
            # "hypervolume": hv.calculate(archive),
            "epsilon_indicator": ei.calculate(archive),
            "inverted_gd": ig.calculate(archive),
            "spacing": sm.calculate(archive),
            "nfe": int(nfe),
        }
        metrics.append(scores)
    metrics = pd.DataFrame.from_dict(metrics)
    # sort metrics by number of function evaluations
    metrics.sort_values(by="nfe", inplace=True)
    return metrics


def get_calculate_metrics(problem_model, folder_name, n_seed, no_scenario):

    merge_archs = []
    metrics = []
    archives = []
    for i in range(n_seed):
        arch = ArchiveLogger.load_archives(
            os.path.join(folder_name, f'results_optimize_box_{no_scenario}_max_{i}.tar.gz'))
        for key in arch.keys():
            arch[key] = arch[key].drop(columns='Unnamed: 0')
            arch[key] = arch[key].rename(columns={x: 'z_' + "_".join(x.split(" ")) for x in arch[key].columns})
            arch[key] = arch[key].rename(columns={x: x.replace('.', 'z') for x in arch[key].columns})
        merge_archs.extend([val for val in arch.values()])
        archives.append(arch)
    reference_set = epsilon_nondominated(merge_archs, [0.025] * len(problem_model.outcomes),
                                             to_problem(problem_model, searchover="levers"))
    for arch in archives:
        metric = calculate_metrics(arch, reference_set, problem_formulation_id)
        metrics.append(metric)
    return metrics, reference_set


def plot_metrics(metrics, convergences):
    sns.set_style("white")
    fig, axes = plt.subplots(nrows=5, figsize=(8, 12), sharex=True)
    ax1, ax2, ax3, ax4, ax5 = axes
    for metric, convergence in zip(metrics, convergences):
        # ax1.plot(metric.nfe, metric.hypervolume)
        # ax1.set_ylabel("hypervolume")
        ax1.plot(convergence.nfe, convergence.epsilon_progress)
        ax1.set_ylabel("$\epsilon$ progress")
        ax2.plot(metric.nfe, metric.generational_distance)
        ax2.set_ylabel("generational distance")
        ax3.plot(metric.nfe, metric.epsilon_indicator)
        ax3.set_ylabel("epsilon indicator")
        ax4.plot(metric.nfe, metric.inverted_gd)
        ax4.set_ylabel("inverted generational\ndistance")
        ax5.plot(metric.nfe, metric.spacing)
        ax5.set_ylabel("spacing")
        ax5.set_xlabel("nfe")
    sns.despine(fig)


if __name__ == '__main__':
    problem_formulation_id = 8
    model = dummy_model(problem_formulation_id)

    folder = 'experiment_5'
    seed_num = 2
    scenario_num = 1

    metrics, reference_set = get_calculate_metrics(model, folder, seed_num, scenario_num)

    convergences = [pd.read_csv(os.path.join(
        folder, f'convergence_results_optimize_box_{scenario_num}_max_{i}.csv'))
        for i in range(seed_num)]

    plot_metrics(metrics, convergences)
    plt.show()
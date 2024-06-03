import os
import functools

import numpy as np
import pandas as pd

from ema_workbench import (Model, MultiprocessingEvaluator, Policy, Scenario, Samplers, ScalarOutcome,
                           ema_logging, load_results, save_results)
from ema_workbench.em_framework import (sample_uncertainties, ArchiveLogger)

from problem_formulation import get_model_for_problem_formulation

def compare_with_threshold(data, operator="ABOVE", threshold=0):
    """
    return the portion of elements in data in comparison with threshold
    :param data: 1-D array
    :param operator: one of the following ["ABOVE", "BELOW", "EQUAL"]
    :param threshold: threshold to compare with
    :type data: 1-D array
    :type operator: str
    :type threshold: float
    :return: the portion of elements in data in comparison with threshold
    """
    if operator == "ABOVE":
        return np.sum(data > threshold)/data.shape[0]
    elif operator == "BELOW":
        return np.sum(data < threshold)/data.shape[0]
    elif operator == "EQUAL":
        return np.sum(data == threshold)/data.shape[0]
    else:
        raise ValueError("Operator must be among [ABOVE, BELOW, EQUAL]. But get {}".format(operator))
        return

def partial_object_threshod(operator="ABOVE", threshold=0):
    """
    Return a partial object of the compare_with_threshold function
    :param operator: one of the following ["ABOVE", "BELOW", "EQUAL"]
    :param threshold: threshold to compare with
    :type operator: str
    :type threshold: float
    :return: the partial object of the compare_with_threshold function
    """
    return functools.partial(compare_with_threshold, operator=operator, threshold=threshold)

def partial_object_percentile(q=10):
    """
    Return a partial object of the np.percentile function with q percentiles
    :param q: the percentile param to the np.percentile function
    :type q: int
    :return: the partial object of the np.percentile function
    """
    return functools.partial(np.percentile, q=q)

if __name__ == "__main__":
    problem_formulation_id = 6
    dike_model, planning_step = get_model_for_problem_formulation(problem_formulation_id)

    n_scenarios = 10 #range [200, 500]
    scenarios = sample_uncertainties(dike_model, n_scenarios)

    annual_damage_percentile = 50
    investment_cost_percentile = 75
    expected_death_threshold = 0
    robustness_functions = [
        ScalarOutcome(
            "%d percentile annual damage" % annual_damage_percentile,
            kind=ScalarOutcome.MINIMIZE,
            variable_name="A.3_Expected Annual Damage",
            function=partial_object_percentile(annual_damage_percentile)
        ),
        ScalarOutcome(
            "death below %d" % expected_death_threshold,
            kind=ScalarOutcome.MINIMIZE,
            variable_name="A.3_Expected Number of Deaths",
            function=partial_object_threshod(operator="ABOVE", threshold=expected_death_threshold)
        ),
        ScalarOutcome(
            "%d percentile investment cost" % investment_cost_percentile,
            kind=ScalarOutcome.MINIMIZE,
            variable_name="A.3_Dike Investment Costs",
            function=partial_object_percentile(investment_cost_percentile),
        )
    ]

    nfe = int(100) #range [5000 - 50000]
    with MultiprocessingEvaluator(dike_model) as evaluator:
        robust_results = evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            nfe=nfe,
            epsilons=[0.025,] * len(robustness_functions),
        )

    pd.DataFrame(scenarios.designs, columns=scenarios.params).to_csv(os.path.join('experiment', 'scenarios.csv'), index=False) #write scenarios to csv
    robust_results.to_csv(os.path.join('experiment', 'robust_policies.csv'), index=False) # write policies to csv

"""
Run experiments using:
    100 randomly generated scenarios. These scenarios were input to the robust_optimize() to identify the most robust policies.
    the robust policies that the robust_optimize() function produce.

The output of the experiments will be used in sensitivity analysis to identify favorable policies.
"""

import os
import time
import random
import pandas as pd

from ema_workbench import save_results
from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario, Samplers

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging

from problem_formulation import get_model_for_problem_formulation

if __name__ == '__main__':
    problem_formulation_id = 7
    dike_model, planning_step = get_model_for_problem_formulation(problem_formulation_id)

    scenario_dict = pd.read_csv(os.path.join('experiment', 'scenarios(100_1000).csv'))\
        .loc[:, [u.name for u in dike_model.uncertainties]].to_dict('records')
    scenarios = [Scenario(i, **scenario) for (i, scenario) in enumerate(scenario_dict)]

    # # with policies
    policy_dict = pd.read_csv(os.path.join('experiment', 'robust_policies(100_1000).csv'))\
                      .loc[:, [l.name for l in dike_model.levers]].to_dict('records')
    policies = [Policy(i, **policy) for (i, policy) in enumerate(policy_dict)]
    for i in range(5):
        random.seed(i)
        with MultiprocessingEvaluator(dike_model) as evaluator:
            results = evaluator.perform_experiments(scenarios=scenarios, policies=policies)

        save_results(results, os.path.join('experiment', 'results_pol%d_prb%d_sd%d.tar.gz' % (len(policies), problem_formulation_id, i)))

    # without policies
    zero_policy = {"DaysToThreat": 0}
    zero_policy.update({f"DikeIncrease {n}": 0 for n in planning_step})
    zero_policy.update({f"RfR {n}": 0 for n in planning_step})
    pol0 = {}

    for key in dike_model.levers:
        s1, s2 = key.name.split("_")
        pol0.update({key.name: zero_policy[s2]})

    pol0 = Policy("Policy 0", **pol0)
    for i in range(5):
        random.seed(i)
        with MultiprocessingEvaluator(dike_model) as evaluator:
            results = evaluator.perform_experiments(scenarios=scenarios, policies=pol0)

        save_results(results, os.path.join('experiment', 'results_pol%d_prb%d_sd%d.tar.gz' % (0, problem_formulation_id, i)))


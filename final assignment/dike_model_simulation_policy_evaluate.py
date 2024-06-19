"""
Run experiments using:
    100 randomly generated scenarios. These scenarios were input to the robust_optimize() to identify the most robust policies.
    the robust policies that the robust_optimize() function produce.

The output of the experiments will be used in sensitivity analysis to identify favorable policies.
"""

import os
import random
import pandas as pd

from ema_workbench import save_results
from ema_workbench import MultiprocessingEvaluator, Policy, Scenario, Samplers

from problem_formulation import get_model_for_problem_formulation


if __name__ == '__main__':
    problem_formulation_id = 8
    dike_model, planning_step = get_model_for_problem_formulation(problem_formulation_id)

    # reference scenarios
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "ID flood wave shape": 4,
        "planning steps": 2,
    }
    reference_values.update({f"discount rate {n}": 3.5 for n in planning_step})
    scen = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split("_")
        if len(name_split) == 1:
            scen.update({key.name: reference_values[key.name]})
        else:
            scen.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario("reference", **scen)


    for policy in range(5):
        # with policies
        policy_df = pd.read_csv(os.path.join('mordm', f'reference_set_scenario_{policy}.csv'))
        policy_df.columns = policy_df.columns.str.replace('z_', '').str.replace('z', '.')\
                                            .str.replace('e_', 'e ').str.replace('R_', 'R ')
        policy_dict = policy_df.loc[:, [l.name for l in dike_model.levers]].to_dict('records')
        policies = [Policy(i, **policy) for (i, policy) in enumerate(policy_dict)]

        n_seed = 5
        n_scenarios = 50
        for i in range(n_seed):
            random.seed(i)
            with MultiprocessingEvaluator(dike_model) as evaluator:
                results = evaluator.perform_experiments(scenarios=n_scenarios, policies=policies)

            save_results(results, os.path.join('mordm', f'results_policy_{policy}_seed_{i}.tar.gz'))

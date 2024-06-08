"""
Run evaluator.optimize() to find the most robust policies for 5 scenarios that are unfavorable to Dike Ring 3.

The outcomes of the experiments will be used for sensitivity analysis on levers and scenario discovery to identify the most favorable policies for Dike Ring 3.
"""

import os
import time
import random
import pandas as pd

from ema_workbench import save_results
from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario, Samplers, Constraint
from ema_workbench.em_framework.optimization import ArchiveLogger, EpsilonProgress

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging

from problem_formulation import get_model_for_problem_formulation

if __name__ == '__main__':
    problem_formulation_id = 6
    dike_model, planning_step = get_model_for_problem_formulation(problem_formulation_id)

    # generate scenarios
    n_scenarios = 5
    scenarios = pd.read_csv(os.path.join('experiment', 'scenario_cart.csv')).rename(columns={'Unnamed: 0': 'uncertainties'}).set_index('uncertainties')
    scenarios = scenarios.iloc[:, :min(n_scenarios, len(scenarios))]
    scenarios = scenarios.to_dict('dict')
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

    # set constraints
    threshold_death = 3
    threshold_damage = 1.7e9
    threshold_investment = 10e9
    constraints = [Constraint("max expected number of death", outcome_names="A.3_Expected Number of Deaths",
                              function=lambda x: max(0, x - threshold_death)),
                   Constraint("max expected annual damage", outcome_names="A.3_Expected Annual Damage",
                              function=lambda x: max(0, x - threshold_damage)),
                   Constraint("max expected dike investment cost", outcome_names="A.3_Dike Investment Costs",
                              function=lambda x: max(0, x - threshold_investment))]

    n_seed = 5
    nfe = 10000
    # find most favorable policies for each scenario
    for box, scenario in scenarios.items():

        # create ref_scenario for optimize() for each scenario
        for uncertainty, value in scenario.items():
            scen.update({uncertainty: value})
        ref_scenario = Scenario(box, **scen)

        # run optimize() for different seed
        for seed in range(n_seed):
            random.seed(seed)
            convergence_metrics = [ArchiveLogger(
                "experiment",
                [l.name for l in dike_model.levers],
                [o.name for o in dike_model.outcomes],
                base_filename=f"results_optimize_{box}_{seed}.tar.gz",
            ),
                EpsilonProgress(),
            ]
            with MultiprocessingEvaluator(dike_model) as evaluator:
                results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                          reference=ref_scenario,
                                                          epsilons=[0.025] * len(dike_model.outcomes),
                                                          convergence=convergence_metrics,
                                                          constraints=constraints)

            print(f'Complete run: {box} {seed}')





import random

from docplex.mp.model import Model


# baseline model takes nb of kids as argument

def build_zoo_mincost_model(nbKids):
    mdl = Model(name='buses')
    nbbus40 = mdl.integer_var(name='nbBus40')
    nbbus30 = mdl.integer_var(name='nbBus30')
    costBus40 = 500.0
    costBus30 = 400.0
    mdl.add_constraint(nbbus40 * 40 + nbbus30 * 30 >= nbKids, 'kids')
    mdl.minimize(nbbus40 * costBus40 + nbbus30 * costBus30)
    return mdl


# sample absents
nb_kids = 300
max_floating = 30
nb_samples = 50
samples = [random.randint(-max_floating, max_floating) for _ in range(nb_samples)]


def build_sampled_model(**kwargs):
    nrun = kwargs.pop('run', -1)
    nb_floating = samples[nrun % nb_samples]
    print(f"-- running kids model with {nb_floating} floating kids")
    return build_zoo_mincost_model(300 + nb_floating)


if __name__ == "__main__":
    from examples.modeling.multiprocessing.process_pool import run_model_process_pool

    samples = [random.randint(-max_floating, max_floating) for _ in range(nb_samples)]
    allres = run_model_process_pool(model_build_fn=build_sampled_model, nb_process=nb_samples, verbose=False)
    mean_cost = sum(allres) / nb_samples
    print(f"* monte carlo, #samples={nb_samples}, max. absents={max_floating}, mean cost is {mean_cost}")
    print(allres)

#
#
#
# nbKids = 300
# nbSamples = 20
# nbMaxKidsAbsent = 30
#
# nbKidsLess = [random.randint(0, nbMaxKidsAbsent) for i in range(0, nbSamples)]
# nbKidsOptions = [nbKids - nbKidsLess[i] for i in range(0, nbSamples)]
#
# # Monte Carlo optimization
#
# totalCost = 0.0;
# for i in range(0, nbSamples):
#     cost = mincost(nbKidsOptions[i])
#     totalCost += cost
#     print("if we need to bring ", nbKidsOptions[i], " kids  to the zoo");
#     print("cost = ", cost)
#
# print()
# averageCost = 1 / nbSamples * totalCost
#
# print("------------------------------");
# print("average cost = ", math.ceil(averageCost));

"""

which gives



average cost =  3665 

So the school knows 3665 is the figure they could use instead of 3825

4% saving in the budget in that toy example.

But the conclusion is Monte Carlo optimization is pretty simple and helps do more with less. 

"""

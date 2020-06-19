import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

#
# This example shows how to use ProcessPoolExecutor with DOcplex.
# The executor will run a Runner class qwhich will set a different run _index_ at each run.
# You must provide a function which will modify the model with this index.
#
# Here are the following steps:
# 1. wrap the build of the model to be run in the pool executor in a function with only **kwargs arguments
#    for example build_lifegame(**kwargs).
#
#    This function should expect a `run` parameter from 0 to nb_process-1, identifying the run.
#    It should apply variants to the model from this run index.
#    The `run` argument should not be passed to the Model() constructor, as it is not recognized.
#    ** Note**: this build function should not contain any lambdas, as the model is pickled at some point.
#                do not use lambdas for naming variables, use named functions defined at file level
#
# 2. Call run_process_pool with the following arguments:
#       - model_build_fn: the model building function
#       - nb_processes: the number of processes to run
#
# This function returns a list of results, one for each process.
# A result is either None,
# or a dictionary of variable names to values, # with a special key '_objective_value' holding the objective value.


class ModelRunner(object):
    """ A wrapper class for pickling"""

    run_kw = 'run'

    @staticmethod
    def make_result(result, sol):
        # temporary, for now we cannot pickle solutions.
        if sol:
            if result == 'solution':
                return sol
            elif result == 'dict':
                sol_d = sol.as_name_dict()
                sol_d['_objective_value'] = sol.objective_value
                return sol_d
            else:
                # default is objective
                return sol.objective_value
        else:
            return None

    def __init__(self, buildfn, result="objective", verbose=True):
        self.buildfn = buildfn
        self._result = result
        self.verbose = bool(verbose)

    def __call__(self, **kwargs):
        try:
            nrun_arg = kwargs.get(self.run_kw, -1)
            nrun = int(nrun_arg)

        except (KeyError, TypeError):
            print(f"warning: no run number was found in kwargs")
            nrun = -1

        # use the model build function to create one instance
        m = self.buildfn(**kwargs)
        assert m is not None
        mname = m.name
        if self.verbose:
            print('--> begin run #{0} for model {1}'.format(nrun, mname))
        m.name = '%s_%d' % (mname, nrun)

        sol = m.solve()
        if sol:
            timed = m.solve_details.time
            if self.verbose:
                print(
                    '<-- end run #{0} for model {1}, obj={2}, time={3:.2f}s'.format(nrun, m.name, sol.objective_value, timed))
            return self.make_result(self._result, sol)
        else:
            print("*** model {0} has no solution".format(m.name))
            return None


def run_model_process_pool(model_build_fn, nb_process, max_workers=3,
                           result='objective', verbose=True):
    """ Runs N models in parallel, with variants from one run to the other

    :param model_build_fn: the function to build the baseline model instance,
        signature musy be (**kwargs) -> Model
    :param nb_process: number of processes, an integeer, assumed to be >= 1
    :param max_workers: maximum number of workers
    :param result: a string indicating which kind of result is to be returned.
        Default is 'objective' to return the value of the objective.
        Other possible values are: 'dict' to return a dictionary of variable names to values.
    :param verbose: a flag to print info messages.

    :return: the list of solve results.
        For now, the result of a solve is either None, if solve failed,
         of a dict of variable names to values.
    """
    if nb_process <= 2:
        raise ValueError(f"Expecting a number of processes >= 2, {nb_process} was passed")
    pool_runner = ModelRunner(model_build_fn, result=result, verbose=verbose)
    allres = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_i = {executor.submit(pool_runner, run=i): i for i in range(nb_process)}
        # executor.shutdown(wait=False)
        for future in concurrent.futures.as_completed(future_to_i):
            res = future.result()
            if res is not None:
                allres.append(res)
            else:
                return None
    return allres



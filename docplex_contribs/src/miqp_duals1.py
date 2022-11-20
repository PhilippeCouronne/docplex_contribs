from docplex.mp.model import Model

from contextlib import contextmanager


def build_miqp1(**kwargs):
    qpx1 = Model(name="qpx1", **kwargs)
    x = qpx1.integer_var_list(keys=3, ub=40, name="x")
    qpx1.add(-x[0] + x[1] + x[2] <= 20)
    qpx1.add(x[0] - 3 * x[1] + x[2] <= 30)
    # the quadratic objective itself.
    qobj = x[0] + 2 * x[1] + 3 * x[2]\
           - (33 * x[0] ** 2 + 22 * x[1] ** 2 + 11 * x[2] ** 2 - 12 * x[0] * x[1] - 23 * x[1] * x[2]) / 2
    qpx1.maximize(qobj)
    return qpx1


@contextmanager
def model_qsolvefixed(mdl):
    """ This contextual function is used to temporarily change the type of the model
    to "fixed_MIQP".
    As a contextual function, it is intended to be used with the `with` construct, for example:

    >>> with model_qsolvefixed(mdl) as mdl2:
    >>>     mdl2.solve()

    The  model returned from the `with` has a temporary problem type set to "solveFixex overriding the
    actual problem type.
    This function expects an instance of MIQP model, which has been successfully solved;
    it returns the same model, with its CPLEX type modified to "fixed_MIQP", which can be solved as a QP.
    with all discrete values fixed to their solution value in the first solve.
    when exiting the with block, the actual problem type is restored.

    :param mdl: an instance of `:class:Model`, an integer quadratic problem.

    :return: the same model, with overridden problem type.

    Note:
        - an exception is raised if the model is not quadratic
        - an exception is raised if the model has not been successfully solved
        - QP models are returned unchanged, as this function has no use.
    """

    if not mdl.is_quadratic():
        mdl.fatal("Model_qsolvefixed is valid ony for quadratic models")

    cpx = mdl._get_cplex(do_raise=True, msgfn=lambda: "model_solvefixed requires CPLEX runtime")

    # save initial problem type, to be restored.
    saved_problem_type = cpx.get_problem_type()
    if saved_problem_type in {5, 10}:  # 5 is QC/ 10 is QCP, no integers
        mdl.warning("Model {0} is a QP/QCP model, qsolvefixed does nothing".format(mdl.name))
        return mdl

    if mdl.solution is None:
        # a solution is required.
        mdl.fatal(f"model_solvefixed requires that the model has been solved successfully")
    try:
        cpx.set_problem_type(8)  # 3 is constant fixed_MILP
        yield mdl
    finally:
        cpx.set_problem_type(saved_problem_type)


qm = build_miqp1()

qs1 = qm.solve(log_output=True)

print(f"-- before fixing , problem type is now: {qm._get_cplex_problem_type()}")

with model_qsolvefixed(qm) as fixed_qm:
    print(f"-- after fixing , problem type is now: {fixed_qm._get_cplex_problem_type()}")
    s2 = fixed_qm.solve(log_output=True)
    assert s2
    for lc in fixed_qm.iter_linear_constraints():
        print(f"-- constraint: {lc}, index #{lc.index}, dual={lc.dual_value}, slack={lc.slack_value}")



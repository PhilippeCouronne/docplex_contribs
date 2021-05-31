# This file sjows how to print number of cuts used by type.h


from cplex._internal._subinterfaces import CutType

def get_cut_stats(mdl):
    """ Computes a dicitonary of cut name -> number of cuts used

    Args:
        mdl: an instance of `docplex.mp.Model`

    Returns:
        a dictionary of string -> int,, from cut type to number used (nonzero).
        Unused cut types ar enot mentioned

    Example:
        For delivered model "nurses"
        # {'cover': 88, 'GUB_cover': 9, 'flow_cover': 6, 'fractional': 5, 'MIR': 9, 'zero_half': 9, 'lift_and_project': 5}

    """
    cut_stats = {}
    cpx = mdl.cplex
    cut_type_instance = CutType()
    for ct in cut_type_instance:
        num = cpx.solution.MIP.get_num_cuts(ct)
        if num:
            cutname = cut_type_instance[ct]
            cut_stats[cutname] = num

    return cut_stats

if __name__ == "__main__":
    from examples.delivery.modeling.nurses import build
    nm = build()
    ns = nm.solve(log_output=True)
    cutstats = get_cut_stats(nm)
    print(cutstats)
    print(f"-- total #cuts = {sum(nk for _, nk in cutstats.items())}")
#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: docplex_gencb.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Solve a facility location problem with cut callbacks or lazy constraints.

Given a set of locations J and a set of clients C, the following model is
solved:

 Minimize
  sum(j in J) fixedCost[j]*used[j] +
  sum(j in J)sum(c in C) cost[c][j]*supply[c][j]
 Subject to
  sum(j in J) supply[c][j] == 1                    for all c in C
  sum(c in C) supply[c][j] <= (|C| - 1) * used[j]  for all j in J
              supply[c][j] in {0, 1}               for all c in C, j in J
                   used[j] in {0, 1}               for all j in J

In addition to the constraints stated above, the code also separates
a disaggregated version of the capacity constraints (see comments for the
cut callback) to improve performance.

Optionally, the capacity constraints can be separated from a lazy
constraint callback instead of being stated as part of the initial model.

See the usage message for how to switch between these options.
"""

import sys
import traceback

import cplex

# this mixin class contains code to convert a docplex linear constraint
# into a triplet  (lhs, sense, rhs) suitable for use in callbacks.
from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin


class FacilityCallback(object):
    """Callback function for the facility location problem.

    This callback can do three different things:
       - Generate disaggregated constraints algorithmically
       - Generate disaggregated constraints looking through a table
       - Generate capacity constraint as lazy constraints.

    Everything is setup in the invoke function that
    is called by CPLEX.
    """

    def __init__(self, clients, locations, used, supply, eps=1e-6):
        self.clients = clients
        self.locations = locations
        self.used = used
        self.supply = supply
        self.cutlhs = None
        self.cutrhs = None
        self.all_cut_cts = []
        # compute list of indices:
        self.used_indices = [u.index for u in used]
        self.supply_indices = {(c, l): s.index for (c, l), s in supply.items()}
        self.eps = eps
        self.invoke_count = 0
        self.model = self.used[0].model

    def disaggregate(self, context):
        """Separate the disaggregated capacity constraints.

        In the model we have for each location j the constraint

        sum(c in clients) supply[c][j] <= (nbClients-1) * used[j]

        Clearly, a client can only be serviced from a location that is
        used, so we also have a constraint

        supply[c][j] <= used[j]

        that must be satisfied by every feasible solution. These
        constraints tend to be violated in LP relaxation. In this
        callback we separate them.
        """
        EPS = self.eps

        for j in self.locations:
            for c in self.clients:
                s, o = context.get_relaxation_point(
                    [self.supply_indices[c, j], self.used_indices[j]])
                if s > o + EPS:
                    cutmanagement = cplex.callbacks.UserCutCallback.use_cut.purge
                    user_cut = self.supply[c, j] <= self.used[j]
                    print(f'-- {self.invoke_count}: adding user cut : {str(user_cut)}')
                    cut_lhs, cut_sense, cut_rhs = ModelCallbackMixin.linear_ct_to_cplex(user_cut)
                    context.add_user_cut(
                        cut=cut_lhs,
                        sense=cut_sense, rhs=cut_rhs,
                        cutmanagement=cutmanagement,
                        local=False)

    def cuts_from_table(self, context):
        """Generate disaggregated constraints looking through a table."""
        EPS = self.eps
        # for lhs, rhs in zip(self.cutlhs, self.cutrhs):
        #     # Compute activity of left-hand side
        #     act = sum(c * x for c, x in zip(lhs.val,
        #                                     context.get_relaxation_point(lhs.ind)))
        #     if act > rhs + EPS:
        #         print('Adding %s [act = %f]' % (str(lhs), act))
        #         cutmanagement = cplex.callbacks.UserCutCallback.use_cut.purge
        #         context.add_user_cut(cut=lhs, sense="L", rhs=rhs,
        #                              cutmanagement=cutmanagement, local=False)
        for cut in self.all_cut_cts:
            cpx_lhs, cpx_sense, cpx_rhs = ModelCallbackMixin.linear_ct_to_cplex(cut)
            act = sum(c * x for c, x in zip(cpx_lhs,
                                            context.get_relaxation_point(cpx_lhs.ind)))
            if act > cpx_rhs + EPS:
                print('Adding %s [act = %f]' % (str(cut), act))
                cutmanagement = cplex.callbacks.UserCutCallback.use_cut.purge
                context.add_user_cut(cut=cpx_lhs, sense=cpx_sense, rhs=cpx_rhs,
                                     cutmanagement=cutmanagement, local=False)

    def lazy_capacity(self, context):
        """Generate capacity constraint as lazy constraints."""
        if not context.is_candidate_point():
            raise Exception('Unbounded solution')
        EPS = self.eps
        for j in self.locations:
            isused = context.get_candidate_point(self.used_indices[j])
            served = sum(context.get_candidate_point(
                [self.supply_indices[c, j] for c in self.clients]))
            if served > (len(self.clients) - 1.0) * isused + EPS:
                lazyct = self.model.sum(self.supply[c, j] for c in self.clients) <= ( (len(self.clients) - 1.0) * self.used[j])
                print(f'-- adding lazy constraint %s <= %d*used[%d]' %
                      (' + '.join(['supply[%d,%d]' % (x, j) for x in self.clients]),
                       len(self.clients) - 1, j))

                print(f"-- adding lazy constraint: {lazyct}")
                lz_lhs, lz_sense, lz_rhs = ModelCallbackMixin.linear_ct_to_cplex(lazyct)
                context.reject_candidate(constraints=[lz_lhs], senses=lz_sense, rhs=[lz_rhs])
                # context.reject_candidate(
                #     constraints=[cplex.SparsePair([self.supply_indices[c, j] for c in self.clients] +
                #                                   [self.used[j].index],
                #                                   [1.0] * len(self.clients) + [-(len(self.clients) - 1)]), ],
                #     senses='L',
                #     rhs=[0.0, ])

    def invoke(self, context):
        """Whenever CPLEX needs to invoke the callback it calls this
        method with exactly one argument: an instance of
        cplex.callbacks.Context.
        """
        self.invoke_count += 1
        print(f"--- call {self.__class__.__name__}.invoke #{self.invoke_count}")
        try:
            if context.in_relaxation():
                if self.cutlhs:
                    self.cuts_from_table(context)
                else:
                    self.disaggregate(context)
            elif context.in_candidate():
                self.lazy_capacity(context)
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


# endif

# default data
DEFAULT_FIXED_COSTS = [480, 200, 320, 340, 300]
DEFAULT_SUPPLY_COSTS = [[24, 74, 31, 51, 84],
                        [57, 54, 86, 61, 68],
                        [57, 67, 29, 91, 71],
                        [54, 54, 65, 82, 94],
                        [98, 81, 16, 61, 27],
                        [13, 92, 34, 94, 87],
                        [54, 72, 41, 12, 78],
                        [54, 64, 65, 89, 89]]


# def admipex8(datadir, from_table, lazy, use_callback):
#     """Solve a facility location problem with cut callbacks or lazy
#     constraints.
#     """
#     # Read in data file. The data we read is
#     # fixedcost  -- a list/array of facility fixed cost
#     # cost       -- a matrix for the costs to serve each client by each
#     #               facility
#
#     # pylint: disable=unbalanced-tuple-unpacking
#     fixedcost, cost, _ = read_dat_file(datadir + '/' + 'facility.dat')
#
#     # Create the model
#     locations = list(range(len(fixedcost)))
#     clients = list(range(len(cost)))
#     cpx = cplex.Cplex()
#     # Create variables.
#     # - used[j]      If location j is used.
#     # - supply[c][j] Amount shipped from location j to client c. This is a
#     #                number in [0,1] and specifies the percentage of c's
#     #                demand that is served from location i.
#     # Note that we also create the objective function along with the variables
#     # by specifying the objective coefficient for each variable in the 'obj'
#     # argument.
#     used = cpx.variables.add(obj=fixedcost,
#                              lb=[0] * len(locations), ub=[1] * len(locations),
#                              types=['B'] * len(locations),
#                              names=['used(%d)' % (j) for j in locations])
#     supply = [cpx.variables.add(obj=[cost[c][j] for j in locations],
#                                 lb=[0] * len(locations), ub=[1] * len(locations),
#                                 types=['B'] * len(locations),
#                                 names=['supply(%d)(%d)' % (c, j) for j in locations])
#               for c in clients]
#
#     # The supply for each client must sum to 1, i.e., the demand of each
#     # client must be met.
#     cpx.linear_constraints.add(
#         lin_expr=[cplex.SparsePair(supply[c], [1.0] * len(supply[c]))
#                   for c in clients],
#         senses=['E'] * len(clients),
#         rhs=[1.0] * len(clients))
#
#     # Capacity constraint for each location. We just require that a single
#     # location cannot serve all clients, that is, the capacity of each
#     # location is nbClients-1. This makes the model a little harder to
#     # solve and allows us to separate more cuts.
#     if not lazy:
#         cpx.linear_constraints.add(
#             lin_expr=[cplex.SparsePair(
#                 [supply[c][j] for c in clients] + [used[j]],
#                 [1.0] * len(clients) + [-(len(clients) - 1.0)])
#                 for j in locations],
#             senses=['L'] * len(locations),
#             rhs=[0] * len(locations))
#
#     # Tweak some CPLEX parameters so that CPLEX has a harder time to
#     # solve the model and our cut separators can actually kick in.
#     cpx.parameters.mip.strategy.heuristicfreq.set(-1)
#     cpx.parameters.mip.cuts.mircut.set(-1)
#     cpx.parameters.mip.cuts.implied.set(-1)
#     cpx.parameters.mip.cuts.gomory.set(-1)
#     cpx.parameters.mip.cuts.flowcovers.set(-1)
#     cpx.parameters.mip.cuts.pathcut.set(-1)
#     cpx.parameters.mip.cuts.liftproj.set(-1)
#     cpx.parameters.mip.cuts.zerohalfcut.set(-1)
#     cpx.parameters.mip.cuts.cliques.set(-1)
#     cpx.parameters.mip.cuts.covers.set(-1)
#
#     # Setup the callback.
#     # We instantiate the callback object and attach the necessary data
#     # to it.
#     # We also setup the contexmask parameter to indicate when the callback
#     # should be called.
#     facilitycb = FacilityCallback(clients, locations, used, supply)
#     contextmask = 0
#     if use_callback:
#         contextmask |= cplex.callbacks.Context.id.relaxation
#         if from_table:
#             # Generate all disaggregated constraints and put them into a
#             # table that is scanned by the callback.
#             facilitycb.cutlhs = [cplex.SparsePair([supply[c][j].index, used[j].index],
#                                                   [1.0, -1.0])
#                                  for j in locations for c in clients]
#             facilitycb.cutrhs = [0] * len(locations) * len(clients)
#     if lazy:
#         contextmask |= cplex.callbacks.Context.id.candidate
#
#     # Callback is setup attach it to the model
#     if contextmask:
#         cpx.set_callback(facilitycb, contextmask)
#     cpx.write('model.lp')
#     cpx.solve()
#
#     print('Solution status:                   %d' % cpx.solution.get_status())
#     print('Nodes processed:                   %d' %
#           cpx.solution.progress.get_num_nodes_processed())
#     print('Active user cuts/lazy constraints: %d' %
#           cpx.solution.MIP.get_num_cuts(cpx.solution.MIP.cut_type.user))
#     tol = cpx.parameters.mip.tolerances.integrality.get()
#     print('Optimal value:                     %f' %
#           cpx.solution.get_objective_value())
#     values = cpx.solution.get_values()
#     for j in [x for x in locations if values[used[x]] >= 1 - tol]:
#         print('Facility %d is used, it serves clients %s' %
#               (j, ', '.join([str(x) for x in clients
#                              if values[supply[x][j]] >= 1 - tol])))


def build_supply_model(fixed_costs, supply_costs, lazy=False, use_callback=False, **kwargs):
    from docplex.mp.model import Model
    m = Model(name='suppy', **kwargs)

    nb_locations = len(fixed_costs)
    nb_clients = len(supply_costs)
    range_locations = range(nb_locations)
    range_clients = range(nb_clients)

    # --- Create variables. ---
    # - used[l]      If location l is used.
    # - supply[l][c] Amount shipped from location j to client c. This is a real
    #                number in [0,1] and specifies the percentage of c's
    #                demand that is served from location l.
    used = m.binary_var_list(range_locations, name='used')
    supply = m.binary_var_matrix(range_clients, range_locations, name='supply')
    m.used = used
    m.supply = supply
    # --- add constraints ---
    # The supply for each client must sum to 1, i.e., the demand of each
    # client must be met.
    m.add_constraints(m.sum(supply[c, l] for l in range_locations) == 1 for c in range_clients)
    # Capacity constraint for each location. We just require that a single
    # location cannot serve all clients, that is, the capacity of each
    # location is nbClients-1. This makes the model a little harder to
    # solve and allows us to separate more cuts.
    if not lazy:
        m.add_constraints(
            m.sum(supply[c, l] for c in range_clients) <= (nb_clients - 1) * used[l] for l in range_locations)

    # Setup the callback.
    # We instantiate the callback object and attach the necessary data
    # to it.
    # We also setup the contexmask parameter to indicate when the callback
    # should be called.
    facilitycb = FacilityCallback(range_clients, range_locations, used, supply, eps=1e-6)
    contextmask = 0
    if use_callback:
        contextmask |= cplex.callbacks.Context.id.relaxation
        # Generate all disaggregated constraints and put them into a
        # table that is scanned by the callback.
        facilitycb.all_cut_cts = [supply[c,l] <= used[l] for l in range_locations for c in range_clients]
    if lazy:
        contextmask |= cplex.callbacks.Context.id.candidate

    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
    params = m.parameters
    params.threads = 1
    params.mip.strategy.heuristicfreq = -1
    params.mip.cuts.mircut = -1
    params.mip.cuts.implied = -1
    params.mip.cuts.gomory = -1
    params.mip.cuts.flowcovers = -1
    params.mip.cuts.pathcut = -1
    params.mip.cuts.liftproj = -1
    params.mip.cuts.zerohalfcut = -1
    params.mip.cuts.cliques = -1
    params.mip.cuts.covers = -1

    # --- set objective ---
    # objective is to minimize total cost, i.e. sum of location fixed cost and supply costs
    total_fixed_cost = m.dot(used, fixed_costs)
    m.add_kpi(total_fixed_cost, 'Total fixed cost')
    total_supply_cost = m.sum(supply[c, l] * supply_costs[c][l] for c in range_clients for l in range_locations)
    m.add_kpi(total_supply_cost, 'Total supply cost')
    m.minimize(total_fixed_cost + total_supply_cost)

    # Callback is setup attach it to the model
    if contextmask:
        m.cplex.set_callback(facilitycb, contextmask)

    return m


if __name__ == "__main__":
    spm = build_supply_model(DEFAULT_FIXED_COSTS, DEFAULT_SUPPLY_COSTS, lazy=True, use_callback=True)
    spm.print_information()
    sps = spm.solve(log_output=False)
    assert sps is not None
    assert abs(sps.objective_value - 843) <= 0.5


# --- call FacilityCallback.invoke #1
# -- adding lazy constraint supply[0,0] + supply[1,0] + supply[2,0] + supply[3,0] + supply[4,0] + supply[5,0] + supply[6,0] + supply[7,0] <= 7*used[0]
# -- adding lazy constraint: supply_0_0+supply_1_0+supply_2_0+supply_3_0+supply_4_0+supply_5_0+supply_6_0+supply_7_0 <= 7used_0
# -- adding lazy constraint supply[0,1] + supply[1,1] + supply[2,1] + supply[3,1] + supply[4,1] + supply[5,1] + supply[6,1] + supply[7,1] <= 7*used[1]
# -- adding lazy constraint: supply_0_1+supply_1_1+supply_2_1+supply_3_1+supply_4_1+supply_5_1+supply_6_1+supply_7_1 <= 7used_1
# -- adding lazy constraint supply[0,2] + supply[1,2] + supply[2,2] + supply[3,2] + supply[4,2] + supply[5,2] + supply[6,2] + supply[7,2] <= 7*used[2]
# -- adding lazy constraint: supply_0_2+supply_1_2+supply_2_2+supply_3_2+supply_4_2+supply_5_2+supply_6_2+supply_7_2 <= 7used_2
# -- adding lazy constraint supply[0,3] + supply[1,3] + supply[2,3] + supply[3,3] + supply[4,3] + supply[5,3] + supply[6,3] + supply[7,3] <= 7*used[3]
# -- adding lazy constraint: supply_0_3+supply_1_3+supply_2_3+supply_3_3+supply_4_3+supply_5_3+supply_6_3+supply_7_3 <= 7used_3
# --- call FacilityCallback.invoke #2
# -- 2: adding user cut : supply_3_1 <= used_1
# -- 2: adding user cut : supply_0_2 <= used_2
# -- 2: adding user cut : supply_5_2 <= used_2
# -- 2: adding user cut : supply_6_3 <= used_3
# -- 2: adding user cut : supply_1_4 <= used_4
# -- 2: adding user cut : supply_2_4 <= used_4
# -- 2: adding user cut : supply_4_4 <= used_4
# -- 2: adding user cut : supply_7_4 <= used_4
# --- call FacilityCallback.invoke #3
# -- 3: adding user cut : supply_5_0 <= used_0
# -- 3: adding user cut : supply_1_1 <= used_1
# -- 3: adding user cut : supply_7_1 <= used_1
# -- 3: adding user cut : supply_2_2 <= used_2
# -- 3: adding user cut : supply_4_2 <= used_2
# -- 3: adding user cut : supply_0_4 <= used_4
# -- 3: adding user cut : supply_3_4 <= used_4
# -- 3: adding user cut : supply_6_4 <= used_4
# --- call FacilityCallback.invoke #4
# -- 4: adding user cut : supply_1_3 <= used_3
# --- call FacilityCallback.invoke #5
# -- 5: adding user cut : supply_1_0 <= used_0
# --- call FacilityCallback.invoke #6
# -- 6: adding user cut : supply_6_1 <= used_1
# --- call FacilityCallback.invoke #7
# -- 7: adding user cut : supply_2_1 <= used_1
# --- call FacilityCallback.invoke #8
# -- 8: adding user cut : supply_0_1 <= used_1
# --- call FacilityCallback.invoke #9
# --- call FacilityCallback.invoke #10
# --- call FacilityCallback.invoke #11
# -- 11: adding user cut : supply_5_0 <= used_0
# -- 11: adding user cut : supply_0_2 <= used_2
# --- call FacilityCallback.invoke #12
# -- adding lazy constraint supply[0,4] + supply[1,4] + supply[2,4] + supply[3,4] + supply[4,4] + supply[5,4] + supply[6,4] + supply[7,4] <= 7*used[4]
# -- adding lazy constraint: supply_0_4+supply_1_4+supply_2_4+supply_3_4+supply_4_4+supply_5_4+supply_6_4+supply_7_4 <= 7used_4
# --- call FacilityCallback.invoke #13
# -- adding lazy constraint supply[0,1] + supply[1,1] + supply[2,1] + supply[3,1] + supply[4,1] + supply[5,1] + supply[6,1] + supply[7,1] <= 7*used[1]
# -- adding lazy constraint: supply_0_1+supply_1_1+supply_2_1+supply_3_1+supply_4_1+supply_5_1+supply_6_1+supply_7_1 <= 7used_1
# --- call FacilityCallback.invoke #14
# -- 14: adding user cut : supply_3_3 <= used_3
# --- call FacilityCallback.invoke #15
# -- 15: adding user cut : supply_3_0 <= used_0
# --- call FacilityCallback.invoke #16
# -- 16: adding user cut : supply_7_0 <= used_0
# --- call FacilityCallback.invoke #17
# -- 17: adding user cut : supply_0_3 <= used_3
# --- call FacilityCallback.invoke #18
# -- 18: adding user cut : supply_0_0 <= used_0
# --- call FacilityCallback.invoke #19
# -- 19: adding user cut : supply_7_3 <= used_3
# --- call FacilityCallback.invoke #20
# -- 20: adding user cut : supply_5_1 <= used_1
# --- call FacilityCallback.invoke #21
# --- call FacilityCallback.invoke #22
# --- call FacilityCallback.invoke #23
# -- 23: adding user cut : supply_2_2 <= used_2
# --- call FacilityCallback.invoke #24
# --- call FacilityCallback.invoke #25
# -- 25: adding user cut : supply_6_2 <= used_2
# --- call FacilityCallback.invoke #26
# -- 26: adding user cut : supply_7_2 <= used_2
# --- call FacilityCallback.invoke #27
# -- 27: adding user cut : supply_3_2 <= used_2
# --- call FacilityCallback.invoke #28
# -- 28: adding user cut : supply_4_3 <= used_3
# --- call FacilityCallback.invoke #29
# --- call FacilityCallback.invoke #30
# --- call FacilityCallback.invoke #31
# -- 31: adding user cut : supply_5_4 <= used_4
# --- call FacilityCallback.invoke #32
# -- 32: adding user cut : supply_5_3 <= used_3
# --- call FacilityCallback.invoke #33
# -- 33: adding user cut : supply_6_0 <= used_0
# --- call FacilityCallback.invoke #34
# -- 34: adding user cut : supply_2_0 <= used_0
# --- call FacilityCallback.invoke #35
# -- 35: adding user cut : supply_2_3 <= used_3
# --- call FacilityCallback.invoke #36
# --- call FacilityCallback.invoke #37
# --- call FacilityCallback.invoke #38
# --- call FacilityCallback.invoke #39
# --- call FacilityCallback.invoke #40
# --- call FacilityCallback.invoke #41
# --- call FacilityCallback.invoke #42
# --- call FacilityCallback.invoke #43
# --- call FacilityCallback.invoke #44
# -- 44: adding user cut : supply_4_0 <= used_0
# --- call FacilityCallback.invoke #45
# --- call FacilityCallback.invoke #46
# --- call FacilityCallback.invoke #47
# --- call FacilityCallback.invoke #48
# --- call FacilityCallback.invoke #49
# --- call FacilityCallback.invoke #50
# --- call FacilityCallback.invoke #51
# -- 51: adding user cut : supply_4_1 <= used_1
# --- call FacilityCallback.invoke #52
# --- call FacilityCallback.invoke #53
# --- call FacilityCallback.invoke #54
# --- call FacilityCallback.invoke #55
# --- call FacilityCallback.invoke #56
# --- call FacilityCallback.invoke #57
# --- call FacilityCallback.invoke #58
# --- call FacilityCallback.invoke #59
# --- call FacilityCallback.invoke #60
# --- call FacilityCallback.invoke #61
# --- call FacilityCallback.invoke #62
# --- call FacilityCallback.invoke #63
# --- call FacilityCallback.invoke #64
# --- call FacilityCallback.invoke #65
# --- call FacilityCallback.invoke #66
# -- 66: adding user cut : supply_1_2 <= used_2

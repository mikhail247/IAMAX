#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

from docplex.cp.solution import CpoSolveResult
from docplex.cp.solver.solver_listener import CpoSolverListener


class MonitorRelativeDarkPermissionReduction(CpoSolverListener):
    """
    Callback that stops search if the reduction of dark permissions is below the relative threshold
    specified by the parameter dark_permission_tolerance.
    """

    def __init__(self, dark_permission_tolerance):
        self.dark_permission_tolerance = dark_permission_tolerance

    def result_found(self, solver, solution):
        from src.iam_optimization_cp import get_relative_dark_permissions_fraction

        if not isinstance(solution, CpoSolveResult):
            return

        relative_dark_permissions_fraction = get_relative_dark_permissions_fraction(solution.model, solution)
        print("MonitorRelativeDarkPermissionReduction: status: {} rel_dark_perm_fraction: {}".format(
            solution.get_solve_status(), relative_dark_permissions_fraction))

        if relative_dark_permissions_fraction is not None \
                and relative_dark_permissions_fraction < self.dark_permission_tolerance:
            print("MonitorRelativeDarkPermissionReduction: search aborted "
                  "due to rel_dark_perm_fraction {} falling below the tolerance level {}"
                  .format(relative_dark_permissions_fraction, self.dark_permission_tolerance))
            solver.abort_search()

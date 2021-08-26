#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2003  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from __future__ import division

import operator
from builtins import object
from math import exp

import pyre
from pyre.handbook.units.pressure import atm

R = pyre.handbook.constants.fundamental.gas_constant

_refP = atm / R


class Equilibrium(object):
    def __init__(self, reagents, dim):
        self._reagents = reagents
        self.dimPhaseSpace = dim
        return

    def __call__(self, T):
        K_p = self._equilibriumConstant(T)
        K_c = K_p * (_refP / T) ** (self.dimPhaseSpace)
        return K_c

    def _equilibriumConstant(self, T):
        deltaGibbs = 0.0
        for species, coefficient in self._reagents:
            deltaGibbs += coefficient * species.g_RT(T)

        return exp(-deltaGibbs)


# version
__id__ = "$Id$"

#
# End of file
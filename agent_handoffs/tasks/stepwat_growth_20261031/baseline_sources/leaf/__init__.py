'''leaf: vegetation side of the coupled native soil-water / individual-plant growth model.

`leaf.coupling.CoupledSimulation` composes the water-free vegetation `Simulation`
(benchmarks.daily_water_growth, vegetation_only=True) with the native layered column
(maple_syrup.subsurface) and the explicit transpiration withdrawal operator
(maple_syrup.subsurface_transpiration). Native imports are lazy: importing `leaf` alone
does not require the native package.
'''
from leaf.coupling import CoupledSimulation, CoupledPhaseError  # noqa: F401

__all__ = ['CoupledSimulation', 'CoupledPhaseError']

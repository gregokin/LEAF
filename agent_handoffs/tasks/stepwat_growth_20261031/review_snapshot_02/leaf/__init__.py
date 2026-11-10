'''leaf: vegetation side of the coupled native soil-water / individual-plant growth model.

`leaf.coupling.CoupledSimulation` composes the water-free vegetation `Simulation`
(benchmarks.daily_water_growth, vegetation_only=True) with the native layered column
(maple_syrup.subsurface) and the explicit transpiration withdrawal operator
(maple_syrup.subsurface_transpiration). Native imports are lazy: importing `leaf` alone
does not require the native package.

`leaf.stepwat` holds the OPT-IN STEP-style individual aboveground growth adaptation
(CoupledSimulation(..., stepwat_traits=...) / Simulation(..., vegetation_only=True, stepwat_traits=...));
without that keyword the legacy fixture growth and all legacy defaults are numerically unchanged.
'''
from leaf.coupling import CoupledSimulation, CoupledPhaseError  # noqa: F401
from leaf import stepwat  # noqa: F401  (pure numpy/numba module; no native import)

__all__ = ['CoupledSimulation', 'CoupledPhaseError', 'stepwat']

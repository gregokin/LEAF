import numpy as np
from maple_syrup import subsurface as ss
from leaf import CoupledSimulation

case = dict(name="example", nx=8, ny=8, dx=0.25, grass_count=10, shrub_count=1,
            grass_diameter=0.15, shrub_diameter=2.0, root_multiplier=3.0,
            pattern="uniform", seed=1, layers=3, max_edges=100000, max_bytes=100000000)
layers = ss.soil_layers([0, 0.1, 0.3, 1.0], theta_residual=0.05,
                       theta_field_capacity=0.25, theta_saturation=0.45,
                       diffusivity_m2_s=2e-7, theta_evaporation_stress=0.20)
column = ss.prepare_column(layers, np.ones(64, dtype=bool), 0.25**2, backend="numba")
soil_state = ss.uniform_state(column, [0.15, 0.15, 0.15])
sim = CoupledSimulation(case, column, soil_state)
added_water_m = np.zeros((3, 64), dtype=np.float64)
soil_evaporation_m_s = np.full(64, 3e-8, dtype=np.float64)
for day in range(7):
    sim.step_day(added_water_m, soil_evaporation_m_s, phase="interevent")
soil_state = sim.state
assert sim.summary()["ok"]

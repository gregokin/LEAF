# Approved case specification and source facts

This is an INPUT-AUTHORING task, not permission to execute LEAF, MAPLE, SYRUP, MASY, plant/landscape generators, validators, tests or GPU work. Read/Edit/Write only. No new driver or source-code changes. The user requests a fully visible, annotated example case, including a year of alternating wind/water events. Be precise about existing native formats versus planning manifests. Do not claim an unimplemented end-to-end case runner exists.

## Case and reference landscape

Use `cases/jornada_grass05_shrub10_256/`. All paths in case manifests resolve from THIS case directory; document that convention. The latest user vegetation replaces that of the reference MAPLE case:

- grass `BOER` (Bouteloua eriopoda proxy): cover 0.05, diameter mean 0.10 m, SD 0.02 m.
- shrub `PROGLA` (Prosopis glandulosa proxy): cover 0.10, diameter mean 1.0 m, SD 0.10 m.
- These are USER choices, not measured field estimates. Initial ages 730 and 1825 days respectively are AUTHOR trial assumptions; generated initial seasonal AGB is zero. Aboveground biomass throughout means DRY MASS, excluding roots/tissue water.
- Conditional uniform random centres, positive normal diameters, global nonoverlap initially (`overlap=disallowed`), per-species union cover. This is a hard-core/rejection process, NOT an unconditioned Poisson process after exclusion. No toroidal biology: finite domain, edge-clipped crowns and roots; no invented edge mode. Later growth/recruitment can overlap. No claim exact sample mean/SD/count until generation.
- Expected initial counts from N=f A/[pi/4 (meanD^2+sdD^2)] at 4096 m2: ~25,073 grass, ~516 shrubs. Estimate only, not generated output.

MAPLE reference `/home/okin/MAPLE/cases/leaf_shrubs11_bimodal135_256_12h/{case.yaml,source/generation.json}`: 256 x 256 cells, dx=dy=0.25 m (64 x 64 m), seed1173, 12 initial sediment voxels each0.1 m, flat reference level0.6 m with periodic spectral Gaussian perturbation of EXACT peak-to-peak relief0.05 m, correlation lengths1.0 m. All are reference computational settings, not observed El Paso terrain. `MAPLE/src/maple/case_tools/generators/random_field.py::generate_spectral_gaussian_field` is the existing implementation; do NOT duplicate it. Bare seed1173 goes directly to this function. Store recipe now; future preparation calls actual upstream public generator and adds0.6 m. No surface.npy exists yet. Hydraulic/soil water layers are NOT sediment voxels or active sediment layer.

The reference has periodic TRANSPORT boundaries, but LEAF placement/roots are finite. Flag this as a future coupling choice, not toroidal exclusion already implemented. Do not import the reference case's sediment calibration or plant list; it contains an explicitly test-only calibration irrelevant to this LEAF input request.

## Native JSON and API facts (code context is supplied)

Native generation schema `leaf.generation_config` version1: strict allowed keys only. Use domain nx/ny/dx_m/dy_m/origin_x_m/origin_y_m. Notes may be strings, JSON comments forbidden. Choose seed20261111 (AUTHOR); cover tolerance0.001 absolute fraction (0.1 percentage point), cover_subsamples8, max_iterations12, time_days0, max_plants200000,max_attempts2000000,max_cover_cells4194304,max_cover_bytes8388608,max_working_bytes1048576. These are numerical/resource choices, not science. The two coverage bitmaps at2048x2048 each total8388608 bytes. Source implementation defines exact byte admission; do not promise generation will pass these caps.

Native species catalog `leaf.species_catalog`,version1,units exactly biomass kg,length m,time days; root_layers3. ONE complete record per species, all traits together. Required allometry,roots,porosity,growth,water_use,reproduction,life_expectancy,mortality,disturbance. `notes`/`status`/`references` must obey respective native schemas. Do not add unknown keys into strict traits. See supplied code/examples. Allometry D=a_d B^b_d and H=a_h B^b_h for kg DRY AGB.

APPROVED TRIAL VALUES (not field calibration):

BOER: diameter a=.9,b=.45; height a=.9,b=.3. Initial meanD implies B=.0075763685 kg dry AGB and H~.208m. PROGLA: diameter a=1.6,b=.4;height a=.8,b=.4 (H=D/2 as in MAPLE reference); initial meanD implies B=.3088161778kg. These coefficients are author trial power-law geometry anchors (diameter exponents originate in LEAF example fixtures, shrub geometry matches MAPLE example), NOT published standing-AGB equations. USDA FEIS gives qualitative/height plausibility only. ANPP regressions MUST NOT be substituted for standing biomass.

Roots lateral radius =3 x canopy radius grass,5 x shrub (user's 3-5x range; trial endpoints). Water uptake fractions, shallow->deep layers[0,.2,.5,1.2]m:
BOER [.39639133268704063,.3360854884442371,.26752317886872223]
PROGLA [.33283263377678235,.32719499000870794,.33997237621450965]
Derived from Vito&Sala2026 combined2-year beta .977/.983: fraction=(beta**lo_cm-beta**hi_cm)/(1-beta**120). F(z)=1-beta**z, depth in CENTIMETRES. Conditional normalization over1.2m; omitted tails .0612840765/.1277680127. Tracer absorption is an UPTAKE PROXY, not measured literal root density. Fixed profiles/no plasticity. This fitted-curve integration does not impose a majority shrub uptake below50cm; do not confuse the paper's binned result with these fractions. Both species reach every layer. Lateral density uses shared finite root builder.

Porosity .2 both: uncalibrated constant copied from existing LEAF example; UNUSED in LEAF ecology but consumed by native MAPLE wind. Do not claim universally unused.

Growth `stepwat_requested_support_v1`: BOER mature_biomass_kg .1,intrinsic_rate_per_year .8, live_fraction .6,extra_growth_rate_per_year .1; PROGLA mature_biomass_kg10,intrinsic_rate_per_year .2,live_fraction .4,extra_growth_rate_per_year .05. ALL author hypotheses within native admitted rate0..1/year, structural STEP precedent only. NO295/year fit.
BOER activity [0,0,0,.1,.3,.6,1,1,1,.4,.05,0], temperature min5,opt30,max45 C. PROGLA activity[0,0,0,.4,.8,1,1,1,.8,.4,.05,0], temperatures5,30,45 C. Exact arrays/cardinal temperatures are AUTHOR assumptions inspired by FEIS seasonal descriptions, not measured cardinal fits. Model temperature multiplier gates demand/growth; NO freeze mortality/fire module provided.
Water law BOTH `biomass_specific_pet_v1`. BOER demand_m3_per_kg_day .0234, PROGLA .0104. Units m3 per kg STANDING DRY AGB per day. BOER comes from local descriptive reanalysis of moist-soil savanna grasses (Belovitch2023); PROGLA is conditional velvet-mesquite seedling/pot proxy Peterson&Neofotis2004, NOT adult honey mesquite. Existing repo docs explain limitations. reference_pet_mm_per_day5 BOTH is AUTHOR normalization, not measured experimental/site mean PET. Further scaling by PET/referencePET, monthly activity and temperature is LEAF's chosen adaptation, not a source-calibrated equation.
Support k=live_fraction/(365*q) at reference conditions: .07024938531787846 BOER,.10537407797681772 PROGLA kg*YEAR/m3. These are newly DERIVED support-balance priors, NOT direct biomass-production WUE yields or original joint fits. Annual calendar normalization follows native code. Keep growth fraction U/D applied ONCE, total across roots; no relocation of unmet demand, water owned/debited by SYRUP exactly once.

Mortality law age_background_stress_v1: nominal mean age-component lifespan15yr BOER/80yr PROGLA (AUTHOR, not field means). Weibull shape2, scale mean/Gamma(1.5)=16.92568750643269/90.27033336764101yr; background_hazard_per_year0 for both (isolate explicit age law). Stress metric support_ratio,low_threshold.25 both,exempt_until_age_days365/730,limit_years2/3,counter_cap_years10 both,mortality_probability.25/.1,biomass_loss_fraction.2/.1. Every exact mortality number AUTHOR uncalibrated assumption; STEP/ECOTONE are structural precedents only. `life_expectancy.mean`15/80 unit years is metadata, NOT an age cutoff. Full model lifespan can shorten due to stress/disturbance.

Disturbance mechanical_response_v1: burial net increment,height_fraction,linear; BOER threshold.1,saturation.8,max_injury_fraction.5,lethal_depth_m null,lethal_height_fraction1.0; PROGLA .1,1.0,.5,null,1.5. Exposure net/linear BOER injury_threshold_m.02,saturation.15,max_injury_fraction.5,lethal_depth_m null,dislodgement_depth_m.2; PROGLA .05,.5,.4,null,.6. ALL exact thresholds are AUTHOR testable response hypotheses, not literature coefficients. Abrasion=null, explicitly disabled until dose units/response calibrated; retain documented socket, never equate sediment flux to dose. Establishment GROUND anchor retained forever; no initial plant-height field. Surface->soil layer remapping deferred.

Reproduction simple_seed_clonal_v1. BOER seed=null; clonal min_age_days365,min_agb_dry_kg.01,probability.1,max_daughters_per_parent1,initial_agb_dry_kg.001,max_donation_fraction.1,dispersal_radius_m.25,max_attempts_per_daughter8. PROGLA clonal=null (resprouting not equivalent to independent ramets); seed establishment_probability.05,max_recruits_per_patch1,initial_agb_dry_kg.002,initial_seasonal_fraction0. Species-specific exact numbers AUTHOR trial choices; qualitative grass vegetative/shrub seed precedent FEIS. Native reproduction needs explicit session reproduction_patch_size_m5m, not one trial per hydraulic cell. Annual boundary calls once; new IDs/ages0, parent-funded clonal biomass, externally credited seedling biomass; no seed bank/water filter currently. Generation overlap=disallowed does NOT constrain growth or reproduction later.

## Soil / runtime planning manifests (NOT new native input schemas)

Write annotated TOML with exact intended mappings to public APIs and state explicitly no generic `leaf run case.toml` loader exists. Native species/generation can already be read by LEAF. Full coupled year driver not supplied/qualified here. Do not claim annual_plan.csv is native MASY scenario schema.

SYRUP current branch feature/subsurface-hydrology owns persistent soil. Actual API:
soil_layers(depth_edges_m,theta_residual,theta_field_capacity,theta_saturation,diffusivity_m2_s,theta_evaporation_stress,evaporation_factor,gravity_drainage); prepare_column(layers,active_mask,cell_area,backend,reserve_bytes); uniform_state(column,theta_initial,t_s=0); prepare_transpiration(column). Scalars broadcast to all3layers as currently supported. Proposed depth_edges_m[0,.2,.5,1.2]; theta_residual.065,theta_saturation.41 (Carsel&Parrish sandy-loam texture priors, not site-specific); field_capacity.20,evaporation_stress.15,initialtheta[.10,.12,.14],diffusivity_m2_s2e-7,evaporation_factor1,gravity_drainage true are AUTHOR/existing-API illustrative assumptions. D is NOT Ksat. All65536cellsactive,area.0625m2. No daily extra rain, irrigation or runon in interevent: added_water_m zeros; storms' accepted infiltration ONLY from SYRUP. Keep subsurface storage between storms, no event-final reset. Full evolving texture/voxel remap is future, use uniform sandy-loam coefficients now; shared MASY surface owns geometry and ultimatelytexture.

Potential soil evaporation: set explicitly to 0.7 * shared PET (convert mm/day to m/s), provisional AUTHOR partition, plus native moisture stress. LEAF demands already biomass-specific. This is NOT calibrated canopy/bare energy partition; do not claim E+T<=PET, and do not double debit or call full PET a soil evaporation observation. SYRUP/caller must own & qualify shared atmospheric E/T partition before coupled execution. Native ledger closes WATER, not energy. All E/T/growth disabled during wind, rain AND recession. At event boundaries actual transport handles water; no biology catch-up. Future surface-moisture wind feedback remains disabled socket.

Runtime planning choices: backend cuda (explicit device admission at future run; no fallback); compiled shared root builder; daily accounting at most86400s split further at midnight/growth/events; fixed7-day growth grid and roots after growth/populationchanges; flush at transport start. max_edges20000000,max_bytes2147483648,reserve_bytes67108864; these AUTHOR caps not measured requirements. Annual reproduction max_new_plants20000,max_total_records200000,max_attempts200000. No raise on refusal. rng_seed20261111. Calendar start2027-01-01T00:00:00-07:00,end2028-01-01T00:00:00-07:00; fixed MST noDST, native model epoch0. Observation monthly (user selectable) plus event/final snapshots of SAME canonical CSV; whole restart format3 distinct from CSVformat1. Inputs expected `inputs/surface.npy`,`inputs/plants.csv` after separate generation; no placeholders masquerading as actual plant population. Outputs e.g `outputs/plants_{index:04d}.csv`,`outputs/checkpoints/` user selectable; choose a monthly reporting cadence spec and explain driver/recording mapping, don't invent supported cadence loader.

## Climate and scheduler sources

NOAA/NWS primary https://www.weather.gov/epz/epz_area_climate (1991-2020 El Paso normals):
month, TminF,TmeanF,TmaxF,Pin
1,34.5,46.5,58.6,.39
2,38.9,51.5,64.1,.40
3,45.5,58.7,71.9,.24
4,53.3,66.6,80.0,.17
5,62.1,75.4,88.7,.43
6,70.6,83.9,97.1,.73
7,73.0,84.4,95.8,1.58
8,71.8,82.9,94.0,1.67
9,65.4,76.9,88.3,1.52
10,54.0,66.7,79.4,.59
11,42.0,54.5,67.0,.43
12,34.4,46.1,57.8,.63
Annual8.78in=223.012mm, July-Sep121.158mm (~54.3%). Exact units: C=(F-32)*5/9, mm=25.4*in. Repeat each month's converted Tmin/Tmax for EVERY daily row in2027; no interpolation/no observed year/extremes. Derive arithmetic dailyTmean fromextrema (rounding differs slightly from publishedTmean). User explicitly asks average regionalclimate; no claim exactJornada271.2mm.

Native daily weather format `masy.weather.v1`,weather_id nonempty,utc_offset -07:00,sampling daily, units temperature degC,relative_humidity fraction,vpd kPa,shortwave_radiation MJ_m2_day,pet mm_day; temperature {tmean:arithmetic_mean_of_tmin_tmax,extrema:true}; humidity {supplied:none}; radiation {supplied:false}; pet {method:ecohyd_hargreaves}; provenance {kind:prescribed,description:...,source:...}. rows365 objects {date YYYY-MM-DD,tmin_c,tmax_c}. NO RH or radiation values invented. Hargreaves precedent FAO56Eq52 https://www.fao.org/4/x0490e/x0490e07.htm; runtime uses actual SYRUP weather_pet implementation, not copied formula. No PET computed in this task. latitude31.8 degrees is rounded AUTHOR ElPaso-area coordinate proxy, must be bound in future MAPLEcase.yaml location.latitude_deg for native MASY; not derived from NOAA monthlytable or weather file. Exact modelmeteorology requires later observed/stochastic forcing, freezingextremesnotrepresented.

Annual transport plan:36 wind/rain PAIRS=72transportrows, intereventbiology fills gaps fromJan1throughendyear. Monthly raincounts [2,2,3,3,3,3,4,4,4,3,3,2]. For2/monthdates4,18;3/month4,14,24;4/month3,10,17,24. Rain starts09:00MST. Preceding wind starts09:00thepreviousday;43200s forcing;u*=.6m/s,bimodal22.5/157.5deg TO (135 separation,vector meaneast90) as reference experimental forcing, NOT average windclimate. Can use compactnativewindexample2blocks21600seach, explicitly differs reference600stogglingbutretainsdirections/duration; annualCSVrefs this sameforcingexample. Rain durations21600sJan/Feb/Mar/Nov/Dec,14400sApr/May/Oct,10800sJun/Jul/Aug/Sep. Split eachmonthlyrain TOTALequally intoN withdepths at.001mm precision, remainderlastrow so exactmonthlytotals. ExampleJulyfirst10.033mm/3h=3.344333333333333mm/h. No backgroundrain. These dates/counts/durations/disaggregation/windforce ALL AUTHOR synthetic choices; only monthlyP/Ttotals are normals. Rain native recession_guard_s21600isMAXnotduration; waterblockendsONLYactualSYRUPterminalrecession. Do not silentlytruncate/shiftforcingifoverlap; refuse. Windadditionalsettlementalsoactualelapsednotignored.

Native event examples STRICT nounknownnoteskeys:
wind {schema:masy.wind_event.v1,event_id,wind_forcing:{blocks:[{shear_velocity_m_s,direction_deg,duration_s},...]}}
rain {schema:masy.rain_event.v1,event_id,water_profile_id:trial_sandy_loam,intervals:[{duration_s,intensity_mm_h}],recession_guard_s}
interevent {schema:masy.interevent.v1,event_id,duration_s:86400,processes:{evaporation:true,transpiration:true,growth:true}}
Document snippets in adjacentREADME, not JSONcomments. A native `masy.scenario.v1` gap_policy supports ONLY idle/reject; idle runs NO hydrology/biology. Therefore annualCSV/settingsTOML are explicit DRIVER DESIGN EXAMPLE, not directly loadable nativeyearscenario; interevent records need runtime filling fromactualacceptedtransportendtolaterfixedstart; splitdaily/weekly/eventboundaries. DO NOTinventgap_policy=interevent orpretendfullcouplingqualified. No full MAPLE/SYRUP transport parameter sets are copied; those owners still require qualified profiles andsourcepairs. This package provides all LEAF parameters plus honestschedulerexamples,not wind/sedimentcalibration.

## Bibliographic sources and distinctions

Provide SOURCES.md with primary clickable links and shortcodes; EACHnumeric/scientific/config value needs an origin in nearbycomment/notes or grouped exhaustivePARAMETERS.md table naming JSONpaths/TOMLsections. An author's explicitly chosen scenario value with rationale/implementationcitation is legitimate provenance, NOT a literature measurement. Structural papers never establish exactauthorcoefficients. Coverall mortalities/recruitmentthresholds,resources,seeds,times,IDs/schema choices; group vectors andunits whenclear. Do not overclaim fieldvalidation.

- NOAA URL above (observed climatic normals, noteventschedule).
- Vito&Sala2026 Ecology107e70384 https://doi.org/10.1002/ecy.70384 (beta .977/.983, traceruptake; finite-depthintegrationourderivation).
- Simonin2000 USDAFEISblackgrama https://research.fs.usda.gov/feis/species-reviews/boueri ; longlivedC4,upper25cmrootconcentration,rootsto1.2m,summergrowth,20-60cmstems,vegetativepathway. ExactcurvecoefficientsNOTgiven.
- Steinberg2001 USDAFEISmesquite https://research.fs.usda.gov/feis/species-reviews/progla ; southwestdeciduousseasonality,seedestablishment/deeproots. ExactLEAFcoefficientsNOTgiven.
- Palmquistetal2018 STEPWAT2 https://doi.org/10.1002/ecs2.2394 ; structuralgrowth/resource/demographyprecedent. LEAFisybridaboveground-onlyadaptation,notcompleteSTEPWATreproduction.
- Belovitchetal2023 https://doi.org/10.1111/nph.18909 ; primaryreleaseddata https://github.com/mbelovitch/NPH-MS-2023-42507/tree/862abe68c7a9f1eb6397c36516bca35c9d445298 . Localdocs/biomass-specific-water-use.md gives independentlycuratedmoist-binmedian23.4L/kgdryAGB/day,30individualmeans;notprinteduniversalpapercoefficient,harvestmassnotdailyB.
- Peterson&Neofotis2004 https://doi.org/10.1007/s00442-004-1688-y . Localdocs/study-stepwat-parameters.md conditionalvelvetmesquiteq10.3-10.5L/kgdryAGB/day,twoseedlingsonepotonevirtualplant,AGBallocationassumptions,notadulthoneymesquitecalibration. Use10.4ascentralproxyandlabelasauthorselectionwithsourcebounds.
- Carsel&Parrish1988 https://doi.org/10.1029/WR024i005p00755 ; sandyloamtheta_r.065,theta_s.41. Primaryreproductionhttps://doi.org/10.1002/2014GL059827 (notclaimdiffusivityorFCfromthattable).
- Allenetal1998FAO56chapter3Eq52 linkabove; exactnativeecohyd_hargreavesimplementationnotidenticaltoreparameterizingFAO.
- ExistingLEAFsource/examples,MAPLEreferencerecipe,andAUTHORthiscase2026-11-11forcomputationalanduncalibratedchoices. Userrequestcover/sizes/rootrangeseparatelycitedasUSER.

Keep paraphrases from eachprimarypage concise (avoid longquotes or copyingarticletext). Fullsourceauthorityisthisfactsdocument+boundedcopies; nootherpayloadread/network/tools.

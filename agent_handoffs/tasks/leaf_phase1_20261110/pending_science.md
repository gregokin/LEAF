# Pending scientific questions — no answer assumed

These questions were presented asynchronously during the authorized Phase1
work. As of this record, no answers have arrived. Elapsed time is not approval.
Independent uptake/root/calendar/species/native-water work continues. No
dependent growth/temperature/disturbance law is implemented by the active tasks.

1. **Growth normalization.** The user's exact rule is delivered/requested water
   applied to otherwise planned growth once. The older STEP adapter instead
   determines stress from actual-water biomass support. Recommendation: derive
   STEP-style planned growth using the full current water request, then multiply
   by actual/requested once; separately retain actual support/maintenance deficit
   diagnostics. Alternative: derive the water request itself from STEP support
   requirements. The same shortfall must not enter two growth multipliers.
2. **Phase2 response family.** Burial, erosion/root exposure and abrasion laws
   are unspecified. Recommendation: explicitly configured species-specific
   threshold/response curves with required parameters and uncalibrated examples.
   Alternative: select a specific published disturbance model before coding the
   response functions. Event reporting/establishment-ground sockets already exist.
3. **Daily temperature response.** Recommendation: species-specific minimum,
   optimum and maximum temperatures giving a zero–one–zero piecewise-linear
   modifier, combined with shared scheduler PET and monthly activity; all values
   explicit, no calibrated defaults. Alternative: continue externally supplied
   temperature modifiers while selecting the daily law. The inspected STEPWAT2
   annual-temperature shortcut is not a qualified daily response.

The source precedents and adaptations are recorded in method_notes.md.
Recruitment details remain a later scientific choice; no Phase3 implementation
starts before Phase1 and Phase2 have been completed, validated and published.

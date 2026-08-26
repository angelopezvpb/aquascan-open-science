# AquaScan Phase 0 Field SOP v1

## Pre-Departure
1. Verify AutoCal: run autocal_v1.py; ALL checks must PASS.
2. Charge all batteries (12V deep-cycle + laptop).
3. Print blinded site views (OPERATOR_VIEW only).
4. Verify GNSS RTK base/rover pair functional.
5. Pack equipment per equipment_requirements_v1.csv.
6. Backup laptop — empty data directories.

## Site Arrival (per site)
1. Navigate to site using OPERATOR_VIEW coordinates.
2. Verify site accessible and safe (terrain, weather, hazards).
3. Record: arrival time, weather conditions, site photo.
4. GNSS RTK: set up base station, verify RTK fix (PDOP < 3.0).
5. Survey: record profile origin, end, and extent waypoints.

## ERT Deployment
1. Lay out 48-electrode cable along profile.
2. Insert electrodes at 5m spacing.
3. Record electrode positions via GNSS RTK (±0.1 m target).
4. Connect cable to resistivity meter.
5. Run contact resistance check.
   - GREEN: all < 3 kOhm → proceed.
   - YELLOW: 3–5 kOhm → improve contact, re-check.
   - RED: > 5 kOhm → re-stake, add water/gel, re-check.
   - If > 5 electrodes RED after 3 attempts → abort ERT for this site.
6. Run AutoCal ERT checks (gain, offset, leakage).

## ERT Acquisition
1. Acquire Wenner-Schlumberger array (6 a-levels).
2. After every 20 quadrupoles: one reciprocal measurement.
3. Acquire Dipole-Dipole array (8 n-levels).
4. Monitor: current stability, voltage saturation.
5. If saturation detected → reduce current, re-acquire.
6. At acquisition closure:
   - Compute SHA-256 of ERT_RAW file.
   - Log: hash, timestamp, number of quadrupoles, QC status.

## REPEAT SITE: R1 Immediate Repeat (sites marked repeat)
1. Leave electrodes and cable in place.
2. Wait 5 minutes.
3. Re-run ERT acquisition with identical parameters.
4. Label: ERT_RAW_REPEAT_R1.

## Seismic Deployment
1. Lay out 24-geophone cable along same/slightly-offset profile.
2. Plant geophones at 2m spacing.
3. Record geophone positions via GNSS RTK.
4. Connect to seismograph.
5. Run AutoCal seismic checks (channel response, noise, trigger).
6. Tap-test all geophones: verify all 24 channels respond.

## Seismic Acquisition
1. Define 8 shot positions (end shots + every 3rd geophone).
2. For each shot position:
   a. Position hammer + plate.
   b. Arm trigger.
   c. Strike 5 times, recording each.
   d. Verify trigger time on near-offset trace.
   e. If trigger fails → re-arm, re-strike.
   f. If 3 consecutive trigger failures → flag channel.
3. QC between shots:
   - Check dead/noisy channels.
   - Check first-arrival SNR.
4. At acquisition closure:
   - Compute SHA-256 of SEISMIC_RAW per shot gather.
   - Log: hash, timestamp, shot count, QC status.

## REPEAT SITE: R1 Immediate Repeat
1. Leave geophones in place.
2. Re-acquire critical shot positions (ends + mid-spread).
3. Label: SEISMIC_RAW_REPEAT_R1.

## Geometry Verification
1. Before teardown: GNSS RTK survey of electrode positions.
2. Before teardown: GNSS RTK survey of geophone positions.
3. Record elevation profile along line.

## Data Backup
1. Copy ALL RAW files + metadata + hashes to LOCAL_NVME_1.
2. Verify copies via SHA-256.
3. Copy to LOCAL_NVME_2 (separate device).
4. Verify copies via SHA-256.
5. If network available: upload to CENTRAL_ARCHIVE.
6. Do NOT delete original acquisition-device copies.

## Departure
1. Recover all electrodes, cables, geophones.
2. Inspect equipment for damage.
3. Record: departure time, site issues, equipment status.

## Failure Handling
- Poor electrode contact → re-stake, water, re-check. Max 3 attempts.
- Excessive electrical noise → identify source (power line, fence).
  If unresolvable → flag ACQUIRED_LIMITED.
- Seismic SNR inadequate → increase stacks, check geophone coupling.
  If unresolvable → flag ACQUIRED_LIMITED.
- Trigger failure → re-arm, check switch continuity. Max 5 attempts per shot.
- GNSS unavailable → fall back to tape-measure geometry. Flag GNSS_UNAVAILABLE.
- Incomplete profile → acquire as much as possible. Flag PARTIAL_PROFILE.
- Weather disruption → pause, protect equipment, record interruption time.
- Hardware failure → log component, attempt spare. If catastrophic → flag ACQUISITION_FAILED.

## Expected Logistics
- Sites/day: 2–3 (single), 1–2 (repeat)
- Total field days: 4–6 (including travel + contingency)
- Minimal crew: 2 (operator + assistant)
- Recommended crew: 3 (operator, assistant + safety/QC observer)

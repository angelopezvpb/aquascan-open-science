# Field Hardware and Acquisition Guide

Status: **target BOM only — no hardware has been acquired yet.** Gate 11
physical acceptance is **NOT READY**. This document describes what is needed
to execute the field protocol, how to verify it before deployment, and how the
data must be exported so the public software can read it.

> This is a planning guide, not a purchase order. Prices are indicative
> planning ranges (EUR, 2026), not quotes. Supplier and model names are shown
> for specification comparison only; no endorsement is implied. Confirm
> current price, availability, export formats and lead time with each supplier.

## 1. Minimum field stack

| Subsystem | Minimum configuration | Protocol reference |
|---|---|---|
| ERT | 48 electrodes, 5 m spacing, 235 m profile, 400 V / 500 mA | `protocols/phase0_ert_acquisition_v1.json` |
| Seismic | 24 channels, 14 Hz vertical geophones, 2 m spacing, 8 kg hammer | `protocols/phase0_seismic_acquisition_v1.json` |
| GNSS RTK | Base + rover, cm-level, PDOP < 3.0, explicit elevation type | `protocols/phase0_field_sop_v1.md` |
| Calibration | Known loads + channel/timing checks | `protocols/phase0_autocal_v1.json` |
| Power | 12 V LiFePO4, > 11.5 V under load, one full field day | `protocols/phase0_autocal_v1.json` |
| Compute/storage | Rugged laptop + 2 × 1 TB NVMe, offline capable | `protocols/phase0_field_sop_v1.md` |
| Safety | PPE, first aid, communications, land access | `protocols/phase0_field_sop_v1.md` |

The model itself is a scikit-learn RandomForest. **No GPU is required**; the
field laptop runs acquisition software, not model training.

## 2. Data contract — what each instrument must export

The canonical schemas are the contract. A native instrument export (CSV,
SEG-2, SEG-Y or proprietary) must be converted into these structures. **No
real instrument file has been imported yet**; the import path is a Gate 11
item. The current `DATA_PIPELINE_ACCEPTANCE` is `PASS_WITH_LIMITATIONS` on
synthetic fixtures only.

### ERT

Required per quadrupole (A-B-M-N):

- electrode positions along the profile;
- injected current and measured voltage (RAW, never overwritten);
- apparent resistance, geometric factor, apparent resistivity;
- contact resistance;
- stack count and reciprocal pair ID;
- instrument channel;
- QC flags and SHA-256 of the RAW record.

Schema: `schemas/ert_observation_v0.json`.
Software: `src/aquascan/physical/ert/pipeline.py`.

### Seismic

Required per shot:

- shot ID and source position;
- receiver IDs and receiver positions;
- sample rate (1000 Hz) and record length (500 ms);
- RAW waveforms per receiver;
- trigger timestamp;
- stack index and channel numbers;
- QC flags and SHA-256 of the RAW shot payload.

Schema: `schemas/seismic_observation_v0.json`.
Software: `src/aquascan/physical/seismic/pipeline.py`.

### GNSS

Required per point:

- latitude, longitude and elevation;
- **explicit elevation type**: `ELLIPSOIDAL`, `ORTHOMETRIC` or `LOCAL_Z`;
- PDOP and fix quality;
- timestamp.

The software refuses to mix elevation frames silently; see
`src/aquascan/physical/coords.py`. Never convert unknown values to zero.

### RAW handling

- Keep the instrument's original files untouched.
- Compute SHA-256 at acquisition closure (per site or per shot gather).
- Copy to two independent NVMe devices and verify both copies.
- Do not delete the acquisition device's original files.

## 3. ERT — Electrical Resistivity Tomography

Protocol: **48 electrodes × 5 m** (235 m profile), Wenner-Schlumberger
(6 a-levels) + Dipole-Dipole (5 m, 8 n-levels), injection up to **400 V /
500 mA**, contact resistance **< 5 kΩ** (green < 3 kΩ), reciprocal
measurements every 5th quadrupole, stacking 3–6 cycles.

| Component | Specification | Examples | Indicative cost |
|---|---|---|---|
| Resistivity meter (multi-channel) | 48 channels, up to 400 V injection | Iris Syscal Pro 48, AGI SuperSting R8 | 50–70 k€ |
| Resistivity meter (entry) | 24 channels + external multiplexer | Iris Syscal Kid 24, Lippmann 4point light 10W | 20–30 k€ |
| Multiplexer (if not integrated) | ≥ 48 channels | Relay switch boxes (Iris / Lippmann) | 3–8 k€ |
| Electrodes | 48 stainless-steel stakes + spares + contact gel/water | 60–80 cm stakes | 0.5–1 k€ |
| Cables | 48-takeout multi-core cable (235 m) + jumpers/spares | Multi-channel ERT cable | 2–4 k€ |

**Before purchase, confirm:** 48-channel capability (or 24 + multiplexer),
export of RAW current/voltage per quadrupole, contact-resistance measurement,
reciprocal acquisition support, 5 m takeout spacing, and availability of
spare cables/connectors.

## 4. Seismic — P-wave refraction

Protocol: **24 channels, 14 Hz vertical geophones, 2 m spacing** (46 m
profile), **8 kg sledgehammer + plate**, **8 shot positions** (end shots +
every 3rd geophone), **5 stacks/shot**, hammer-switch trigger, **1000 Hz
sampling, 500 ms record**.

| Component | Specification | Examples | Indicative cost |
|---|---|---|---|
| Seismograph | 24 channels, trigger input, 1000 Hz | Geometrics Geode 24, Seistronix Daqlink III | 15–35 k€ new, 10–20 k€ used |
| Geophones | 24 × 14 Hz vertical + takeout cable + spares | SM-24 class | 1–2 k€ |
| Source | 8 kg sledgehammer + steel/aluminium plate | — | 0.2–0.5 k€ |
| Trigger | Hammer-switch closure + geophone at plate | Break-circuit switch + cable | 0.3–1 k€ |

**Before purchase, confirm:** 24-channel recording, 1000 Hz sampling, 500 ms
record length, trigger input, SEG-2/SEG-Y or CSV export, 14 Hz geophones,
2 m takeout spacing, and spare channels/geophones.

## 5. GNSS RTK

Per-electrode and per-geophone positioning (cm-level xy) and elevation, with
explicit ellipsoidal vs orthometric separation.

| Component | Examples | Indicative cost |
|---|---|---|
| RTK base + rover | Emlid Reach RS3 (value), Trimble R10/R12, Leica GS18 | 2.5–4 k€ (entry) / 12–25 k€ (survey-grade) |

In Spain, the national **ERGNSS** network (IGN) provides free RTK corrections
over NTRIP, so a rover and a mobile data connection may be enough. Confirm
coverage at the field site and the correction service's terms.

**Before purchase, confirm:** RTK fix, PDOP < 3.0, cm-level horizontal,
±0.1 m vertical target, CSV export with explicit elevation type, NTRIP
support, and base/rover or network-RTK workflow.

## 6. Calibration reference box

An in-house calibration fixture used before acquisition to verify: known
resistance loads, channel gain/offset, injected-current verification,
reciprocity reference, and leakage for ERT; channel response, noise, trigger
timing, timing skew, and gain consistency for seismic; plus system checks
(power stability, timestamp integrity, storage integrity, GNSS status).

The functional interface is public (`protocols/phase0_autocal_v1.json`); the
proprietary AutoCal implementation is **not** included in this repository.
Only deterministic, public-domain checks are implemented.

## 7. Power

| Component | Indicative cost |
|---|---|
| 2 × LiFePO4 12 V 100 Ah battery | 0.6–1.5 k€ |
| DC-DC regulators + small inverter (if needed) | 0.3–0.8 k€ |
| Portable solar panel (optional) | 0.3–0.6 k€ |

**Before purchase, confirm:** > 11.5 V under acquisition load, runtime for a
full field day, fuses/breaker protection, and a power-loss test that does not
corrupt RAW data.

## 8. Compute and storage

| Component | Indicative cost |
|---|---|
| Rugged field laptop (e.g., Panasonic Toughbook, Dell Rugged) | 2–4 k€ |
| NVMe SSD (local) + archive drive (dual copy, SHA-256 verified) | 0.3–0.6 k€ |

RAW volumes are small (< 100 MB per site for the full ERT + seismic stack),
so storage capacity is not a constraint. The laptop must run offline, have
the required USB/serial ports, and be able to write two independent copies.

## 9. Accessories and safety

50 m measuring tape, stakes/markers, tool kit, transport cases, weather
protection, gloves / hi-vis / first aid, spare cables and fuses.
Indicative cost: 1–2 k€.

> **Electrical safety:** ERT injects up to **400 V**. Operator training, a
> safety protocol, and site signage are mandatory (see the field SOP in
> `protocols/phase0_field_sop_v1.md`). Confirm land access and permits before
> mobilisation.

## 10. Procurement strategy for the pilot

1. **Rent** the resistivity meter and seismograph for the pilot campaign
   (2–3 weeks). The repeatability experiment (immediate + redeployment
   repeats) and cross-site comparisons require the same instrument on every
   site; renting validates the methodology before committing capital.
2. **Buy now:** GNSS RTK, used geophones, hammer/plate, the calibration
   reference box, rugged laptop, and LiFePO4 batteries (~10–15 k€).
3. Buy the ERT + seismic systems only if the pilot confirms the stack is
   scientifically useful.

### Budget scenarios (indicative)

| Scenario | Total |
|---|---|
| Minimal viable — rent ERT + seismograph; buy GNSS RTK, used geophones, source, rugged laptop, power | 8–15 k€ + rentals |
| Buy entry-level — Syscal Kid / 4point light + multiplexer, used seismograph, entry GNSS | 45–65 k€ |
| Mid-range — Syscal Pro 48 / SuperSting R8, new seismograph, survey-grade GNSS | 90–130 k€ |

Not included: VAT, travel, crew, permits, insurance and contingency. Add a
10–20% contingency for repairs, spares and schedule risk.

## 11. Bench acceptance before field deployment

Do not mobilise until every critical check passes. Gate 11 is **NOT READY**
until these are performed on the actual hardware.

### ERT

- [ ] Electrode N maps to physical channel N (no silent swap).
- [ ] Known resistor load per channel: gain deviation < 1%.
- [ ] Open-circuit offset < 0.1 mV.
- [ ] Injected current vs external meter: deviation < 2%.
- [ ] Leakage/noise floor with no current injection: below instrument spec.
- [ ] Reciprocity reference on a known symmetric array: < 1%.
- [ ] Contact-resistance check and abort threshold verified.
- [ ] RAW export contains current/voltage per quadrupole (not only inversion).

### Seismic

- [ ] All 24 channels respond to tap test.
- [ ] Amplitude within ±10% of median.
- [ ] 2 s no-source noise record; RMS below threshold.
- [ ] Trigger timing within ±0.5 ms across 5 strikes.
- [ ] Timing skew < 1 sample across end channels.
- [ ] Gain consistency < 5% on a common-source gather.
- [ ] RAW export contains waveforms, sample rate, record length and trigger.

### GNSS

- [ ] RTK fix and PDOP < 3.0 at the test site.
- [ ] Correction loss and recovery behaviour documented.
- [ ] Elevation type recorded explicitly (`ELLIPSOIDAL` / `ORTHOMETRIC` / `LOCAL_Z`).
- [ ] Export includes fix quality and timestamp.

### Power, EMC and data

- [ ] Idle, ERT, seismic and compute load measured.
- [ ] Peak demand and battery runtime measured.
- [ ] Voltage stays > 11.5 V during acquisition.
- [ ] Power-loss test does not corrupt RAW data.
- [ ] EMC: seismic noise with ERT off / idle / injecting.
- [ ] EMC: ERT only, seismic active, GNSS + compute active, all active.
- [ ] Storage write + readback + SHA-256 match.
- [ ] Real instrument file imported into the canonical schema (or converter
      documented and tested).
- [ ] Field SOP dry run with the actual crew and timing.

## 12. Field deployment summary

See `protocols/phase0_field_sop_v1.md` for the full step-by-step SOP.

- **Crew:** minimum 2 (operator + assistant); recommended 3 (operator,
  assistant + safety/QC observer).
- **Logistics:** 2–3 sites/day (single), 1–2 sites/day (repeat); 4–6 field
  days including travel and contingency.
- **ERT:** 48 electrodes at 5 m; green < 3 kΩ, yellow 3–5 kΩ, red > 5 kΩ;
  abort if > 5 electrodes remain red after 3 attempts.
- **Seismic:** 24 geophones at 2 m; 8 shot positions; 5 stacks per shot;
  trigger check on every shot.
- **Data:** SHA-256 at closure; two NVMe copies verified; central archive if
  network is available; never delete instrument originals.
- **Safety:** 400 V ERT, trained operator, signage, PPE, first aid, weather
  and land-access checks.

## 13. What is not included

- The proprietary AutoCal implementation (interface only).
- Acquired or tested hardware (none yet).
- Real field validation or groundwater-detection accuracy.
- Binding prices, supplier quotes or lead times.
- A tested real-instrument import path (still a Gate 11 item).

## 14. References

- `protocols/phase0_field_sop_v1.md` — field procedure.
- `protocols/phase0_ert_acquisition_v1.json` — ERT design.
- `protocols/phase0_seismic_acquisition_v1.json` — seismic design.
- `protocols/phase0_autocal_v1.json` — pre-field checks.
- `schemas/ert_observation_v0.json` — ERT data contract.
- `schemas/seismic_observation_v0.json` — seismic data contract.
- `src/aquascan/physical/coords.py` — GNSS/local frame and elevation rules.
- `docs/roadmap.md` — hardware acquisition is the next milestone.

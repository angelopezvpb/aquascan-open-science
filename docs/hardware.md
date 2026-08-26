# Field Hardware — Bill of Materials

Status: **target BOM only — no hardware has been acquired yet.** This document
describes what is needed to execute the field protocol; the acceptance state
is tracked separately (currently: not performed).

Specifications below follow the acquisition protocols in `protocols/`
(`phase0_ert_acquisition_v1.json`, `phase0_seismic_acquisition_v1.json`).
Prices are indicative ranges (EUR, 2026) and vary by supplier, condition
(new/used), and market.

## 1. ERT — Electrical Resistivity Tomography

Protocol: **48 electrodes × 5 m** (235 m profile), Wenner-Schlumberger
(6 n-levels) + Dipole-Dipole (5 m, 8 n-levels), injection up to **400 V /
500 mA**, contact resistance **< 5 kΩ** (green < 3 kΩ), reciprocal
measurements every 5th quadrupole, stacking 3–6 cycles.

| Component | Specification | Examples | Indicative cost |
|---|---|---|---|
| Resistivity meter (multi-channel) | 48 channels, up to 400 V injection | Iris Syscal Pro 48, AGI SuperSting R8 | 50–70 k€ |
| Resistivity meter (entry) | 24 channels + external multiplexer | Iris Syscal Kid 24, Lippmann 4point light 10W | 20–30 k€ |
| Multiplexer (if not integrated) | ≥ 48 channels | Relay switch boxes (Iris / Lippmann) | 3–8 k€ |
| Electrodes | 48 stainless-steel stakes + contact gel/water | 60–80 cm stakes | 0.5–1 k€ |
| Cables | 48-takeout multi-core cable (235 m) + jumpers | Multi-channel ERT cable | 2–4 k€ |

## 2. Seismic — P-wave refraction

Protocol: **24 channels, 14 Hz vertical geophones, 2 m spacing** (46 m
profile), **8 kg sledgehammer + plate**, **8 shot positions** (end shots +
every 3rd geophone), **5 stacks/shot**, hammer-switch trigger, **1000 Hz
sampling, 500 ms record**.

| Component | Specification | Examples | Indicative cost |
|---|---|---|---|
| Seismograph | 24 channels, trigger input | Geometrics Geode 24, Seistronix Daqlink III | 15–35 k€ new, 10–20 k€ used |
| Geophones | 24 × 14 Hz vertical + takeout cable | SM-24 class | 1–2 k€ |
| Source | 8 kg sledgehammer + steel/aluminium plate | — | 0.2–0.5 k€ |
| Trigger | Hammer-switch closure + geophone at plate | Break-circuit switch + cable | 0.3–1 k€ |

## 3. GNSS RTK

Per-electrode and per-geophone positioning (cm-level xy) and elevation, with
explicit ellipsoidal vs orthometric separation.

| Component | Examples | Indicative cost |
|---|---|---|
| RTK base + rover | Emlid Reach RS3 (value), Trimble R10/R12, Leica GS18 | 2.5–4 k€ (entry) / 12–25 k€ (survey-grade) |

Note: in Spain, the national **ERGNSS** network (IGN) provides free RTK
corrections over NTRIP — only a rover and mobile data connection are needed.

## 4. Calibration box (AutoCal)

An in-house calibration fixture used before acquisition to verify: known
resistance loads, channel gain/offset, injected-current verification,
reciprocity reference, and leakage for ERT; channel response, noise, trigger
timing, timing skew, and gain consistency for seismic; plus system checks
(power stability, timestamp integrity, storage integrity, GNSS status).

The functional interface is public (see `protocols/phase0_autocal_v1.json`);
the internal implementation is not included in this repository.

## 5. Power

| Component | Indicative cost |
|---|---|
| 2 × LiFePO4 12 V 100 Ah battery | 0.6–1.5 k€ |
| DC-DC regulators + small inverter (if needed) | 0.3–0.8 k€ |
| Portable solar panel (optional) | 0.3–0.6 k€ |

## 6. Compute & storage

| Component | Indicative cost |
|---|---|
| Rugged field laptop (e.g., Panasonic Toughbook, Dell Rugged) | 2–4 k€ |
| NVMe SSD (local) + archive drive (dual copy, SHA-256 verified) | 0.3–0.6 k€ |

Raw volumes are small (< 100 MB per site for the full ERT + seismic stack),
so storage capacity is not a constraint.

## 7. Accessories & safety

50 m measuring tape, stakes/markers, tool kit, transport cases, weather
protection, gloves / hi-vis / first aid, spare cables and fuses.
Indicative cost: 1–2 k€.

> **Electrical safety:** ERT injects up to **400 V**. Operator training, a
> safety protocol, and site signage are mandatory (see the field SOP in
> `protocols/phase0_field_sop_v1.md`).

## Budget scenarios (indicative)

| Scenario | Total |
|---|---|
| Minimal viable — rent ERT + seismograph for the campaign; buy GNSS RTK, used geophones, source, rugged laptop, power | 8–15 k€ + rentals |
| Buy entry-level — Syscal Kid / 4point light + multiplexer, used seismograph, entry GNSS | 45–65 k€ |
| Mid-range — Syscal Pro 48 / SuperSting R8, new seismograph, survey-grade GNSS | 90–130 k€ |

## Recommendation for Phase 0

1. **Rent** the resistivity meter and seismograph for the pilot campaign
   (2–3 weeks): the repeatability experiment (immediate + redeployment
   repeats) and cross-site comparisons require the same instrument on every
   site, and renting validates the methodology before committing capital.
2. **Buy now**: GNSS RTK, used geophones, hammer/plate, the calibration box
   (in-house design), rugged laptop, and LiFePO4 batteries (~10–15 k€).
3. Buy the ERT + seismic systems only if the pilot confirms the stack is
   scientifically useful.

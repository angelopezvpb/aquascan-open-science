"""Synthetic data generation for testing and development.

Generates synthetic subsurface models and campaign records.

CRITICAL WARNING:
    Synthetic data can be used for:
    - Testing code correctness
    - Pre-training experiments (future)
    - Algorithm development

    Synthetic data CANNOT be used for:
    - Demonstrating real-world performance
    - Validating field capability
    - Claims about actual groundwater detection
"""

from __future__ import annotations

import uuid

import numpy as np

from aquascan.data.models import (
    CampaignRecord,
    GroundTruthClass,
    Provenance,
    SyntheticLayer,
    SyntheticProfile,
)


def generate_multilayer_profile(
    n_layers: int = 3,
    max_depth: float = 100.0,
    water_table_depth: float | None = 30.0,
    noise_level: float = 0.05,
    random_seed: int | None = None,
) -> SyntheticProfile:
    """Generate a synthetic 1D multilayer subsurface profile.

    Args:
        n_layers: Number of subsurface layers.
        max_depth: Maximum depth (m).
        water_table_depth: Depth to water table (m). None for dry profile.
        noise_level: Relative noise level (0-1).
        random_seed: Random seed for reproducibility.

    Returns:
        A SyntheticProfile with layers.
    """
    rng = np.random.RandomState(random_seed)

    # Generate layer boundaries
    boundaries = np.linspace(0, max_depth, n_layers + 1)
    boundaries[1:-1] += rng.uniform(-2, 2, n_layers - 1)
    boundaries = np.sort(boundaries)

    layers = []
    for i in range(n_layers):
        depth_top = boundaries[i]
        depth_bottom = boundaries[i + 1]

        # Layer physically plausible parameters
        # Resistivity: topsoil 50-500, weathered 20-200, bedrock 200-5000
        if i == 0:
            resistivity = rng.uniform(50, 500)
            vp = rng.uniform(300, 800)
        elif i == n_layers - 1:
            resistivity = rng.uniform(200, 5000)
            vp = rng.uniform(1500, 4000)
        else:
            resistivity = rng.uniform(20, 200)
            vp = rng.uniform(800, 2500)

        saturated = water_table_depth is not None and depth_top >= water_table_depth
        if saturated:
            # Saturated layers have lower resistivity and slightly higher Vp
            resistivity *= rng.uniform(0.1, 0.5)
            vp *= rng.uniform(1.1, 1.5)

        layers.append(
            SyntheticLayer(
                depth_top=depth_top,
                depth_bottom=depth_bottom,
                resistivity=resistivity,
                vp=vp,
                saturated=saturated,
            )
        )

    return SyntheticProfile(
        profile_id=str(uuid.uuid4()),
        layers=layers,
        noise_level=noise_level,
        water_table_depth=water_table_depth,
        label="SYNTHETIC",
    )


def generate_synthetic_campaign(
    water_present: bool = True,
    random_seed: int | None = None,
    campaign_id: str | None = None,
) -> CampaignRecord:
    """Generate a single synthetic campaign record.

    Args:
        water_present: Whether the site has groundwater.
        random_seed: Random seed.
        campaign_id: Optional campaign ID.

    Returns:
        A CampaignRecord with synthetic data.
    """
    rng = np.random.RandomState(random_seed)
    cid = campaign_id or f"SYNTH-{uuid.uuid4().hex[:8]}"

    # Synthetic coordinates (random location in Spain area)
    lat = rng.uniform(36.0, 43.5)
    lon = rng.uniform(-9.5, 3.5)

    if water_present:
        water_depth = rng.uniform(10, 80)
        resistivity_mean = rng.uniform(10, 80)  # Low resistivity → water
        vp_mean = rng.uniform(1500, 2500)
        static_wl = water_depth + rng.uniform(-5, 5)
        pump_flow = rng.uniform(0.5, 50.0)
    else:
        water_depth = rng.uniform(80, 200)
        resistivity_mean = rng.uniform(200, 5000)  # High resistivity → dry
        vp_mean = rng.uniform(2500, 5000)
        static_wl = None
        pump_flow = 0.0

    return CampaignRecord(
        campaign_id=cid,
        site_id=f"SYNTH-SITE-{rng.randint(1, 1000)}",
        well_id=f"SYNTH-WELL-{rng.randint(1, 1000)}",
        latitude=lat,
        longitude=lon,
        elevation=rng.uniform(0, 1500),
        geology_domain=rng.choice(["Granite", "Limestone", "Alluvial", "Volcanic"]),
        hydrogeological_unit=rng.choice(["HGU-A", "HGU-B", "HGU-C", "HGU-D"]),
        ert_quality=rng.uniform(0.6, 1.0),
        seismic_quality=rng.uniform(0.6, 1.0),
        ert_resistivity_mean=resistivity_mean,
        ert_resistivity_std=resistivity_mean * rng.uniform(0.1, 0.4),
        seismic_vp_mean=vp_mean,
        seismic_vp_std=vp_mean * rng.uniform(0.05, 0.2),
        water_depth_min=water_depth - 10,
        water_depth_max=water_depth + 10,
        total_depth=rng.uniform(100, 300),
        static_water_level=static_wl,
        pump_test_flow=pump_flow,
        pump_test_duration=rng.uniform(1, 48) if water_present else None,
        water_electrical_conductivity=rng.uniform(200, 3000) if water_present else None,
        water_tds=rng.uniform(100, 2000) if water_present else None,
        ground_truth_score=1.0,  # Synthetic always has perfect GT
        ground_truth_class=GroundTruthClass.A,
        ground_truth_source="synthetic",
        ground_truth_reasons=["SYNTHETIC DATA — perfect ground truth by construction"],
        provenance=Provenance(
            processing_step="generate_synthetic_campaign",
            source_uri="synthetic://generator",
            processing_version="0.1.0",
        ),
        # Note: label is not a CampaignRecord field; it's captured in provenance
    )


def generate_synthetic_dataset(
    n_campaigns: int = 100,
    water_ratio: float = 0.5,
    random_seed: int = 42,
) -> list[CampaignRecord]:
    """Generate a synthetic dataset of campaign records.

    Args:
        n_campaigns: Number of campaigns.
        water_ratio: Fraction of water-positive sites.
        random_seed: Random seed.

    Returns:
        List of CampaignRecord instances.
    """
    rng = np.random.RandomState(random_seed)
    n_water = int(n_campaigns * water_ratio)
    n_dry = n_campaigns - n_water

    records = []
    for i in range(n_water):
        records.append(
            generate_synthetic_campaign(
                water_present=True,
                random_seed=rng.randint(0, 2**31),
                campaign_id=f"SYNTH-WET-{i:04d}",
            )
        )

    for i in range(n_dry):
        records.append(
            generate_synthetic_campaign(
                water_present=False,
                random_seed=rng.randint(0, 2**31),
                campaign_id=f"SYNTH-DRY-{i:04d}",
            )
        )

    # Shuffle
    rng.shuffle(records)
    return records

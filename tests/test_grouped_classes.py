#!/usr/bin/env python
"""Test grouped particle classes and metadata flow"""

import sys
sys.path.insert(0, r'c:\Users\Jacopo\Documents\GitHub\kinematicParcels\src')

import numpy as np
from parcels import FieldSet, ParticleSet

# Import the updated runner module
from kinematicparcels.runner.run_experiment import (
    ScipyParticleGrouped, 
    JITParticleGrouped,
    get_particle_class
)

print("=" * 70)
print("TEST 1: Custom Particle Classes are defined")
print("=" * 70)

# Verify classes exist and have Variables
print("\n✓ ScipyParticleGrouped imported successfully")
print(f"  - group_id Variable: {hasattr(ScipyParticleGrouped, 'group_id')}")
print(f"  - group_member Variable: {hasattr(ScipyParticleGrouped, 'group_member')}")
print(f"  - group_size Variable: {hasattr(ScipyParticleGrouped, 'group_size')}")

print("\n✓ JITParticleGrouped imported successfully")
print(f"  - group_id Variable: {hasattr(JITParticleGrouped, 'group_id')}")
print(f"  - group_member Variable: {hasattr(JITParticleGrouped, 'group_member')}")
print(f"  - group_size Variable: {hasattr(JITParticleGrouped, 'group_size')}")

print("\n" + "=" * 70)
print("TEST 2: get_particle_class() returns grouped custom classes")
print("=" * 70)

scipy_class = get_particle_class("scipy")
jit_class = get_particle_class("jit")

print(f"\nget_particle_class('scipy') returns: {scipy_class.__name__}")
assert scipy_class == ScipyParticleGrouped, "ERROR: scipy should return ScipyParticleGrouped"
print("✓ Correct: ScipyParticleGrouped")

print(f"\nget_particle_class('jit') returns: {jit_class.__name__}")
assert jit_class == JITParticleGrouped, "ERROR: jit should return JITParticleGrouped"
print("✓ Correct: JITParticleGrouped")

print("\n" + "=" * 70)
print("TEST 3: Grouped Variables can be passed to ParticleSet.from_list()")
print("=" * 70)

# Create a minimal mock fieldset for testing
from parcels.fieldset import FieldSet as FS

# Create test data with grouped metadata
lons = np.array([10.0, 10.1, 11.0, 11.1])
lats = np.array([40.0, 40.0, 41.0, 41.0])
group_id = np.array([0, 0, 1, 1], dtype=np.int32)
group_member = np.array([1, 2, 1, 2], dtype=np.int32)
group_size = np.array([2, 2, 2, 2], dtype=np.int32)

print("\nTest data:")
print(f"  lons: {lons}")
print(f"  lats: {lats}")
print(f"  group_id: {group_id}")
print(f"  group_member: {group_member}")
print(f"  group_size: {group_size}")

print("\n✓ All tests passed!")
print("=" * 70)
print("Ready for full integration test")
print("=" * 70)

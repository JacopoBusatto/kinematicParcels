#!/usr/bin/env python
"""Inspect Parcels Variable API"""

from parcels import ScipyParticle, JITParticle, Variable
import numpy as np

# Check what Variables are already defined in ScipyParticle
print("ScipyParticle class attributes:")
scipy_vars = [attr for attr in dir(ScipyParticle) if isinstance(getattr(ScipyParticle, attr), Variable)]
print(f"  Found {len(scipy_vars)} Variable definitions")
for var_name in scipy_vars[:10]:
    var = getattr(ScipyParticle, var_name)
    print(f"  - {var_name}: dtype={getattr(var, 'dtype', 'unknown')}, initial={getattr(var, 'initial', 'unknown')}")

print("\n" + "="*60)
print("Variable initialization signature:")
print("  Variable(name, dtype=..., initial=..., to_write=True)")
print("="*60)

# Test creating a custom class
print("\nCreating custom particle class with group Variables...")

class ScipyParticleGrouped(ScipyParticle):
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)

print("✓ ScipyParticleGrouped created successfully")

class JITParticleGrouped(JITParticle):
    group_id = Variable('group_id', dtype=np.int32, initial=0)
    group_member = Variable('group_member', dtype=np.int32, initial=1)
    group_size = Variable('group_size', dtype=np.int32, initial=1)

print("✓ JITParticleGrouped created successfully")

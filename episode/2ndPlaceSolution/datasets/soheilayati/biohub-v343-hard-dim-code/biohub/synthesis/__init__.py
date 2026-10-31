"""Synthesis engine — manufacture labeled births and faint cells inside real
movies (pre-registered 2026-08-17, Progress/stage-06-scoring-programs.md).

Latent-source principle: events are generated in track space (positions over
frames + lineage) and rendered afterwards by transplanting REAL donor pixels.
Background pixels are never touched; donor signal is added (fluorescence is
additive light). Placement is INTEGER-voxel — matching the integer-voxel GT
annotation convention and preserving donor noise texture exactly (a trilinear
shift would smooth it, which is precisely what a discriminator would catch).
"""

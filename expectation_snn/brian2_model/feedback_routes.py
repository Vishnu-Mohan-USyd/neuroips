"""Fixed-weight feedback H -> V1.

Two routes, topology + signs + weights FIXED (not plastic):
1. H E -> V1 E apical (direct, feature-matched, modulatory), scaled by g_direct.
2. H E -> V1 SOM (feature-linked, suppressive), scaled by g_SOM.

Pre-registered balance sweep with g_total = g_direct + g_SOM held constant:
    r = g_direct / g_SOM in {0.25, 0.50, 1.00, 2.00, 4.00}  (S1..S5)
"""

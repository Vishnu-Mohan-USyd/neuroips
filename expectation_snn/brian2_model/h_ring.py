"""H module: stateful context/prediction ring.

12 channels, 16 LIF E per channel with strong within-channel recurrent excitation
supporting persistent bumps (Wang 2001 PMID 11476885) across 200-500 ms delays.
One broad H inhibitory pool (16 LIF). Cue afferents into H E.

Two disjoint instances:
- H_R: trained on Richter-like leader->trailer sequences (used for Kok, Richter assays).
- H_T: trained on Tang-like rotating-orientation sequences.
"""

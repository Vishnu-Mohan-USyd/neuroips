"""Numerical verification of C3: does the eval_kok.py asymmetry formula
yield POSITIVE for Kok-2012 sharpening (expectation suppresses nonpref
more than pref), and NEGATIVE for the dampening/inversion mechanism
described in the code comment (expectation suppresses pref, enhances
nonpref)?
"""

def asymmetry(pref_exp, pref_unexp, nonpref_exp, nonpref_unexp):
    return (nonpref_unexp - nonpref_exp) - (pref_unexp - pref_exp)


def main():
    # Scenario A — SHARPENING (Kok 2012 as described in literature):
    # Expectation suppresses both but nonpref MORE than pref.
    # Without expectation: pref=1.0, nonpref=0.6
    # With expectation:    pref=0.9, nonpref=0.2 (nonpref collapses more)
    a_sharp = asymmetry(
        pref_exp=0.9, pref_unexp=1.0,
        nonpref_exp=0.2, nonpref_unexp=0.6,
    )
    # Scenario B — DAMPENING as described in code comment:
    # Expectation SUPPRESSES pref, ENHANCES nonpref.
    # Without expectation: pref=1.0, nonpref=0.4
    # With expectation:    pref=0.7, nonpref=0.6 (pref falls, nonpref rises)
    a_damp = asymmetry(
        pref_exp=0.7, pref_unexp=1.0,
        nonpref_exp=0.6, nonpref_unexp=0.4,
    )
    # Scenario C — UNIFORM SUPPRESSION (no preference effect)
    a_unif = asymmetry(
        pref_exp=0.8, pref_unexp=1.0,
        nonpref_exp=0.4, nonpref_unexp=0.5,
    )
    print(f"sharpening_asymmetry={a_sharp:+.3f}")
    print(f"dampening_asymmetry={a_damp:+.3f}")
    print(f"uniform_suppression_asymmetry={a_unif:+.3f}")
    print(f"formula_positive_means_sharpening= {a_sharp > 0}")
    print(f"formula_negative_means_comment_dampening= {a_damp < 0}")


if __name__ == "__main__":
    main()

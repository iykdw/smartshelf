def NatLangFractions(value: int|float, max_value: int|float, margin: float=0) -> str:
    fracs: list[tuple[float, str]] = [(1/5, "a fifth"), (1/4, "a quarter"), (1/3, "a third"), (2/5, "two-fifths"), (1/2, "half-way"), (3/5, "three-fifths"), (2/3, "two-thirds"), (3/4, "three-quarters"), (4/5, "four-fifths")]
    as_frac = value/max_value

    for frac in fracs:
        if frac[0]-margin < as_frac < frac[0]+margin:
            return frac[1]

    return ""

from bisect import bisect_left

def nat_lang_position(val: int, maxval: int) -> str:
    points = {0: "near the left-hand side",
              0.2: "about a fifth in",
              0.25: "about a quarter in",
              0.33: "about a third in",
              0.4: "about two fifths in",
              0.5: "about half-way in",
              0.6: "about three fifths in",
              0.66: "about two thirds in",
              0.75: "about three quarters in",
              0.8: "about four fifths in",
              1: "near the right-hand side"
            }

    fracs = list(points.keys())
    frac_pos = val/maxval

    index = bisect_left(fracs, frac_pos)
    frac = fracs[index]
    return points[frac]

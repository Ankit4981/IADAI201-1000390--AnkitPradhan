"""Verify every text/background colour pair in theme.py meets WCAG AA (4.5:1)."""
from theme import LIGHT, DARK, BADGES, PANEL, PAIRS


def lum(hex_):
    c = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def blend(fg, bg, a):
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(round(a * x + (1 - a) * y) for x, y in zip(f, b))


def main():
    bad = 0
    for name, pal in (("LIGHT", LIGHT), ("DARK", DARK)):
        print(f"--- {name} ---")
        checks = [(f"{f} on {b}", pal[f], pal[b]) for f, b in PAIRS]
        # chart labels sit on the tinted threshold bands (35% opacity over the card)
        for band in ("low", "mod", "high"):
            checks.append((f"muted on chart band {band}", pal["muted"], blend(pal[band], pal["card"], 0.35)))
        for label, fg, bg in checks:
            r = ratio(fg, bg)
            ok = r >= 4.5
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {r:5.2f}  {label}")
    print("--- FIXED (both themes) ---")
    fixed = [(f"badge {k}", v[1], v[0]) for k, v in BADGES.items()]
    fixed += [("panel text", PANEL["text"], PANEL["bg"]), ("panel sub text", PANEL["sub"], PANEL["bg"])]
    for label, fg, bg in fixed:
        r = ratio(fg, bg)
        ok = r >= 4.5
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {r:5.2f}  {label}")
    print("\nALL PASS" if not bad else f"\n{bad} FAILING PAIRS")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

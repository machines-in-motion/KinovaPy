import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KinovaPy.gim4305 import uint_to_float, unpack_reply


def approx(a, b, tol=0.05):
    return abs(a - b) < tol


def main():
    print("=== Testing uint_to_float() ===")
    try:
        # Undoing the README thermometer: 127 in range 0..40 (8 bits) ~ 20.
        got = uint_to_float(127, 0.0, 40.0, 8)
        ok1 = approx(got, 20.0, tol=0.2)
        print(f"{'✅' if ok1 else '❌'} 127 (0..40, 8 bits) -> {got:.2f}, "
              f"expected ~20.0")
    except NotImplementedError:
        print("   uint_to_float() isn't finished yet — open motor/gim4305.py.")
        return

    print("\n=== Testing unpack_reply() ===")
    # This pretend reply was built to mean: id=1, position≈0, velocity≈0,
    # current≈0 (everything sitting in the middle of its range).
    fake_reply = [0x01, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF]
    try:
        state = unpack_reply(fake_reply)
    except NotImplementedError:
        print("   unpack_reply() isn't finished yet — open motor/gim4305.py.")
        return

    checks = []
    checks.append(("id == 1", state.get("id") == 1))
    checks.append(("position ≈ 0", approx(state.get("position", 99), 0.0)))
    checks.append(("velocity ≈ 0", approx(state.get("velocity", 99), 0.0)))
    for name, ok in checks:
        print(f"{'✅' if ok else '❌'} {name}  (got {state})")

    print()
    if ok1 and all(ok for _, ok in checks):
        print("🎉 You can now both SPEAK and LISTEN to the motor. Lesson 5 makes "
              "it move for real!")
    else:
        print("Not quite — check the ❌ lines and your recipe in gim4305.py.")


if __name__ == "__main__":
    main()
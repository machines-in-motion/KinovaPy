import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KinovaPy.gim4305 import float_to_uint, pack_command


def check(name, got, expected):
    ok = (got == expected)
    mark = "✅" if ok else "❌"
    print(f"{mark} {name}: got {got}, expected {expected}")
    return ok


def main():
    print("=== Testing float_to_uint() ===")
    results = []

    # The thermometer example from the README: 20 in the range 0..40 with 8
    # bits should be about halfway, i.e. 127.
    try:
        results.append(check("halfway (20 in 0..40, 8 bits)",
                             float_to_uint(20.0, 0.0, 40.0, 8), 127))
        # Bottom of the range -> 0
        results.append(check("minimum -> 0",
                             float_to_uint(0.0, 0.0, 40.0, 8), 0))
        # Top of the range -> the biggest 8-bit number, 255
        results.append(check("maximum -> 255",
                             float_to_uint(40.0, 0.0, 40.0, 8), 255))
        # A 12-bit halfway check -> ~2047
        results.append(check("12-bit halfway",
                             float_to_uint(0.0, -30.0, 30.0, 12), 2047))
    except NotImplementedError:
        print("   float_to_uint() isn't finished yet — open motor/gim4305.py.")
        return

    print("\n=== Testing pack_command() ===")
    # If you command all zeros, every value sits in the MIDDLE of its range
    # (except kp & kd, whose range starts at 0). Working through the bytes by
    # hand gives this exact result. If yours matches, your bit-packing is right!
    expected = [0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF]
    try:
        data = pack_command(0.0, 0.0, 0.0, 0.0, 0.0)
        pretty_got = " ".join(f"{b:02X}" for b in data)
        pretty_exp = " ".join(f"{b:02X}" for b in expected)
        results.append(check(f"zero-command bytes ({pretty_got})",
                             pretty_got, pretty_exp))
    except NotImplementedError:
        print("   pack_command() isn't finished yet — open motor/gim4305.py.")
        return

    print()
    if all(results):
        print("🎉 ALL CHECKS PASS. You just hand-built a robot command frame!")
        print("   On to Lesson 4 to read the motor's replies.")
    else:
        print("Some checks failed — read the ❌ lines, tweak motor/gim4305.py, "
              "and run me again. You've got this.")



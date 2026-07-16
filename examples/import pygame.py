import pygame
import time

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Using:", joystick.get_name())

while True:
    pygame.event.pump()

    x = joystick.get_axis(0)
    y = joystick.get_axis(1)
    z = joystick.get_axis(2)

    rx = joystick.get_axis(3)
    ry = joystick.get_axis(4)
    rz = joystick.get_axis(5)

    print(
        "Translation:",
        x, y, z,
        "Rotation:",
        rx, ry, rz
    )

    time.sleep(0.05)
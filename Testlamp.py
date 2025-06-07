from gpiozero import OutputDevice
from time import sleep

lamp = OutputDevice(17, active_high=False, initial_value=False)  # GPIO17, aktywacja niskim stanem

print("Włączam lampkę na 3 sekundy...")
lamp.on()
sleep(3)
print("Wyłączam lampkę.")
lamp.off()

#!/usr/bin/env python

import rospy
from diagnostic_msgs.msg import DiagnosticArray
import matplotlib.pyplot as plt

voltage_battery = []
voltage_12v = []
voltage_5v = []

def diagnostic_callback(msg):    
    
    for status in msg.status:
        # print(status)
        if status.name == "jackal_node: Battery":
            battery_volts= None

            for value in status.values:
                if value.key == "Battery Voltage (V)":
                    battery_volts = value.value
                    break
        
            if battery_volts is not None:
                volts = float(battery_volts)
                voltage_battery.append(volts)
                print("Battery Voltage:", volts)
            else:
                print("Battery Voltage not found in the key-value pairs.")
        
        if status.name == "jackal_node: User voltage supplies":
            volts_12v = None
            volts_5v = None

            for value in status.values:
                if value.key == "12V Supply (V)":
                    volts_12v = value.value
                    break

            for value in status.values:
                if value.key == "5V Supply (V)":
                    volts_5v = value.value
                    break

            if volts_12v is not None and volts_5v is not None:
                voltage_12v.append(float(volts_12v))
                voltage_5v.append(float(volts_5v))

            

def listener():
    rospy.init_node('diagnostic_subscriber', anonymous=True)
    rospy.Subscriber("/diagnostics", DiagnosticArray, diagnostic_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
        
    except rospy.ROSInterruptException:
        pass
    
    print(voltage_battery)
    print(voltage_12v)
    print(voltage_5v)

    plt.figure()    

    plt.subplot(3, 1, 1)
    plt.plot(voltage_battery)
    plt.title("Input Voltage: Power Supply")
    plt.ylim([26, 32])
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(voltage_12v)
    plt.title("12V Supply")
    plt.ylim([10, 15])
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(voltage_5v)
    plt.title("5V Supply")
    plt.ylim([3, 8])
    plt.grid()
    
    plt.show()

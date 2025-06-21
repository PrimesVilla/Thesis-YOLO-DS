import time

import serial.tools.list_ports
from click import command

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []

for one in ports:
    portsList.append(str(one))
    print(str(one))

com = input("Select Com port for Arduino #: ")

for i in range(len(portsList)):
    if portsList[i].startswith("COM"+str(com)):
        use = "COM" + str(com)
        print(use)

        serialInst.baudrate = 9600
        serialInst.port = use
        serialInst.open()

while True:
    command = input("Enter command BIO/REC/SPEC/exit: ")
    serialInst.write(command.encode('utf-8'))

    if command == "exit" :
        break


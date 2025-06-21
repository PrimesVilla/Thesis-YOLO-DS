import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

serialInstance = serial.Serial()
portsList = []

for one in ports:
    portsList.append(str(one))
    print(str(one))

com = input("Select Com Port for Arduino by Number: ")

for i in range(len(portsList)):
    if portsList[i].startswith("COM" + str(com)):
        use = "COM" + str(com)
        print(use)

serialInstance.baudrate = 9600
serialInstance.port = use
serialInstance.open()

while (True):
    command = input("Arduino Command (Turn Servo 1 = Type 1 | Turn Servo 2 = Type 2 | Exit = Type 'q': ")
    serialInstance.write(command.encode('utf-8'))

    if command == 'q':
        exit()
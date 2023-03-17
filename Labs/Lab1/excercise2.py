import sys

class Buses:
    busID = ''
    lineID = ''
    xCoor = 0
    yCoor = 0
    time = 0

    def __init__(self, busID='', lineID='', xCoor=0, yCoor=0, time=0):
        self.busID = busID
        self.lineID = lineID
        self.xCoor = xCoor
        self.yCoor = yCoor
        self.time = time




arguments = sys.argv
input_path = arguments[1]
flag = arguments[2]
input_id = arguments[3]

needed_buses = []

f = open(input_path, 'r')
for x in f:
    parameters = x.split()
    bus = Buses(busID = parameters[0], lineID=parameters[1], time=int(parameters[-1]), xCoor=int(parameters[2]), yCoor=int(parameters[3]))
    if flag == '-b':
        if bus.busID == input_id:
            needed_buses.append(bus)
    elif flag == '-l':
        if bus.lineID == input_id:
            needed_buses.append(bus)
f.close()

distance = 0


if flag == '-b':
    for i in range(len(needed_buses)-1):
        distance += abs(needed_buses[i].xCoor - needed_buses[i+1].xCoor)
        distance += abs(needed_buses[i].yCoor - needed_buses[i+1].yCoor)
    print(f"Bus {input_id}: {distance}")
elif flag == '-l':
    speed = 0
    cont = 0
    average = 0
    for i in range(len(needed_buses)-1):
        distance = 0
        if needed_buses[i].busID == needed_buses[i+1].busID:
            distance += abs(needed_buses[i].xCoor - needed_buses[i+1].xCoor)
            distance += abs(needed_buses[i].yCoor - needed_buses[i+1].yCoor)
            time = abs(needed_buses[i+1].time - needed_buses[i].time)
            speed+= distance/time
            #print("distance " + str(distance))
            #print("speed " + str(speed))
            #print("time " + str(time))
            cont += 1

    speed/=cont
    print(f"Line {input_id}: {speed}")
    



import sys

class Person:
    city = ''
    date = ''
        
    def __init__(self, city, date):
        self.city = city
        self.date = date

    def month(self):
        month = self.date
        month = month.split("/")[1]
        return month
            

    def __str__ (self):
        return f"{self.city}: {self.date}"

args = sys.argv
inputPath = args[1]

f = open(inputPath, "r")
births = []

for x in f:
    line = x.split(" ")
    person = Person(line[2], line[-1])
    births.append(person)

cityBirths = {}
monthBirths = {}

for b in births:
    if b.city not in cityBirths:
        cityBirths[b.city] = 1
    else:
        cityBirths[b.city] += 1
    
    if b.month() not in monthBirths:
        monthBirths[b.month()] = 1
    else:
        monthBirths[b.month()] += 1
    
print(monthBirths)


average = len(births) / len(cityBirths.keys())
print(average)

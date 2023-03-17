import sys

class Competitors:
    name = ''
    surname = ''
    country = ''
    scores = []
    final_record = 0

    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        self.scores = scores

    def eliminate_min_max(self):
        min_val = min(self.scores)
        max_val = max(self.scores)
        self.scores.remove(min_val)
        self.scores.remove(max_val)    

    def final_score(self):
        self.eliminate_min_max()
        for s in self.scores:
            self.final_record += s
    
    def country_score(self):
        return {'country': self.country, 'score':self.scores}



arguments = sys.argv
path1 = arguments[1]

f = open(path1, "r")
competiors = []
i = 0
for x in f:
    scores = [] 
    values = x.split(" ")
    for s in values[3:8]:
        scores.append(float (s))
    competiors.append(Competitors(values[0], values[1], values[2], scores))

f.close()
competior_ranks = []
country_rank = {}
for c in competiors:
    c.final_score()
    competior_ranks.append({'name': f"{c.name} {c.surname}", 'score': c.final_record})
    if c.country not in country_rank:
        country_rank[c.country] = c.final_record
    else:
        country_rank[c.country] += c.final_record



competior_ranks.sort(reverse=True, key=(lambda x : x['score']))
sorted_country = sorted(country_rank.items(), key=lambda x: x[1], reverse=True)



for i in range(3):
    print(competior_ranks[i]['name'] + ": " + str(competior_ranks[i]['score']))

print(sorted_country[0])





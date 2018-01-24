from nltk.metrics import *
import wikipedia
import csv
import sys

assigned_class = []
obtained_class = []
path = sys.argv[1]
reader = csv.reader(open(path+'/DataSet_Full.csv', 'r'))

for row in reader:
    #print(row)
    title, category = row
    print("Topic : " + title)
    wiki_pages = wikipedia.page(title)
    wiki_content = str.lower(wiki_pages.content)
    business_count = wiki_content.count("business")
    politics_count = wiki_content.count("politics")
    sports_count =  wiki_content.count("sports")
    technology_count = wiki_content.count("technology")
    travel_count = wiki_content.count("travel")

    max_count = max(business_count, politics_count, sports_count, technology_count, travel_count)

    class_obt = "Business"
    if max_count == politics_count:
        class_obt = "Politics"
    if max_count == technology_count:
        class_obt = "Technology"
    if max_count == travel_count:
        class_obt = "Travel"
    if max_count == sports_count:
        class_obt = "Sports"

    print("Assigned Category : " +category)
    print("Obtained Category : " +class_obt)
    assigned_class.append(category.strip())
    obtained_class.append(class_obt)

accuracy_baseline = accuracy(obtained_class, assigned_class) * 100

print('Accuracy : ' + accuracy_baseline.__str__() + "%")
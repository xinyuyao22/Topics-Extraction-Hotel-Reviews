import json
import csv

filepath = 'C:/Users/xyyao/Documents/GitHub/Topics-Extraction-Hotel-Reviews/hotel-reviews/aspects-annotated-dataset/tripadvisor'

def write_csv(file):
    for line in open(file, 'r', encoding="utf-8"):
        line = json.loads(line)
        for label, sent in zip(line['segmentLabels'],line['segments']):
            if bool(label):
                writer.writerow([sent, list(label.keys())])


with open('aspect_train.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "Aspect"])
    write_csv(filepath + '/train.unique.json')
    write_csv(filepath + '/test.zero.json')
    write_csv(filepath + '/test.one.json')
    write_csv(filepath + '/test.two.json')

with open('aspect_test.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "Aspect"])
    write_csv(filepath + '/test.unique.json')

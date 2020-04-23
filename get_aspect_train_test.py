import json
import csv

filepath = 'C:/NLP/hotel-reviews/outdir/aspects-annotated-dataset/tripadvisor'

def write_csv(file):
    i = 0
    for line in open(file, 'r', encoding="utf-8"):
        print(i)
        line = json.loads(line)
        print(line)
        for label, sent in zip(line['segmentLabels'],line['segments']):
            if bool(label):
                    writer.writerow([sent, str(list(label.keys())[0])])
        i += 1
with open('train_unique.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "Aspect"])
    write_csv(filepath + '/train.unique.json')

with open('test_combined.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "Aspect"])
    write_csv(filepath + '/test.unique.json')
    write_csv(filepath + '/test.zero.json')
    write_csv(filepath + '/test.one.json')
    write_csv(filepath + '/test.two.json')
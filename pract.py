import csv
if __name__ == '__main__':
    reader = csv.reader(open('testTitanic.csv', 'rb'))
    writer = csv.writer(open('newtestTitanic.csv', 'wb'))
    for row in reader:
        row.insert(1,'?')
        writer.writerow(row)

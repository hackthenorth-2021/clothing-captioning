import json
import urllib.request
import csv

with open('train.json') as f:
  data = json.load(f)

f = open('test_all_4.csv', 'a', newline="")
writer = csv.writer(f)

number_images = int(input("How many images to download?"))
start_image = int(input("Start from?"))

for x in range(start_image, number_images):
    #print(data["images"][x])
    #print(data["annotations"][x])
    try:
        #url = data["images"][x]["url"]
        #urllib.request.urlretrieve(url, "images/" + data["images"][x]["imageId"] + '.jfif')
        row = ["gs://imaterialist/" + data["images"][x]["imageId"] + '.jpg']

        for i, value in enumerate(data["annotations"][x]["labelId"]):
            row.append(data["annotations"][x]["labelId"][i])

        writer.writerow(row)
    except:
        print("Error" + str(x))

f.close()

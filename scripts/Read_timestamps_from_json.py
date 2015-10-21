import os, json

os.chdir('/home/hspreeuw/Dropbox/eScienceCenter/Sherlock')

with open('govdocs1.json') as data_file:
    data = json.load(data_file)

i = 0
while True:    
    try:
        item = data["hits"]["hits"][i]['_source']
        try:
            file_createdon = item["file.createdOn"]
            # file_modifiedon = item["file.modifiedOn"]
            # print('file', file_createdon, file_modifiedon)
            print('file', file_createdon)
        except KeyError:
            try:
                folder_createdon = item["folder.createdOn"]                
                print('folder', folder_createdon)
            except KeyError:
                pass  
    except IndexError:
        break
    i += 1

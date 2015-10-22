import os, json, re, datetime
import numpy as np
import pylab as pyl

os.chdir('/home/hspreeuw/Dropbox/eScienceCenter/Sherlock')

with open('cluster-analysis/json/elastic.json') as data_file:
    data = json.load(data_file)
   
yeardigits = 4
startmonth = yeardigits+1
monthdigits = 2
startday = startmonth+monthdigits+1
daydigits = 2   
starthour = startday + daydigits +1
hourdigits = 2

# search_pattern  = '\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d\.\d\d\dZ'
search_pattern  = '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z'

all_dates = np.empty(0)

latestdate = datetime.datetime(2002, 1, 1)
earliestdate = datetime.datetime(2000, 1, 1)

startyear = 1970
startepoch = datetime.datetime(startyear, 1, 1)

secondsperday = 86400.
daysperyear = 365.2422
hoursperday = 24

def datetime_from_match(datetimestring):
    year = int(datetimestring[0:yeardigits])
    month = int(datetimestring[startmonth: startmonth + monthdigits])
    day = int(datetimestring[startday: startday + daydigits])
    hour = int(datetimestring[starthour: starthour + hourdigits])
    # print('year, month, day = ', year, month, day)
    return datetime.datetime(year, month, day, hour)
    
i = 0
while True:    
    try:
        entry = data["hits"]["hits"][i]['_source']
        for k,v in entry.items(): 
            # Decide if v is a single string or a list.
            try:
                #  assert not isinstance(v, str)
                #  assert not isinstance(v, unicode)
                assert hasattr(v, "__iter__")
                for element in v:
                    match = re.search(search_pattern, element)                    
            except AssertionError:
                try:
                    assert isinstance(v, unicode)
                    match = re.search(search_pattern, v)                   
                except AssertionError:
                    try:
                        # This is probably redundant, all strings seems to be unicode.
                        assert isinstance(v, str)                        
                        match = re.search(search_pattern, v)                                      
                    except AssertionError:                  
                        pass
            if match:
                newdate = datetime_from_match(match.group())
                if newdate < latestdate and newdate > earliestdate: 
                    number_of_days = (newdate - startepoch).total_seconds()/(secondsperday * daysperyear)
                    # print('number of days = ', number_of_days)
                    if number_of_days > 0:
                        all_dates = np.append(all_dates, startyear + number_of_days)  
                    else:
                        print('number_of_days, match.group() = ', number_of_days, match.group())                
            match = None
    except IndexError:
        break
    i += 1   
    
print()
print(all_dates.size, all_dates.min(), all_dates.max())

# One bin per hour
num_bins = (all_dates.max() - all_dates.min()) * daysperyear * hoursperday
n, bins, patches = pyl.hist(all_dates, num_bins, normed = False, facecolor='green', alpha=0.5)
pyl.show()             

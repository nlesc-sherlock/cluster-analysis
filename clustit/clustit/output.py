import json
import numpy as np



class OutputCollection(object):

    def __init__(self, data_frame):
        """ Create an Object to collects output from clustit for visualization using DiVE

        :param: data_frame: A data frame containing the output produced by LargeVis
        :type data_frame: pandas.DataFrame

        """

        self.data_frame = data_frame
        self.property_names = []
        self.properties = []


    def to_array(self):
        df = self.data_frame.sort_values('filename')
        return np.array(df[df.columns[1:]], dtype=np.float32)



    def to_DiVE(self, filename=None):
        """ Store the current collection to a JSON file to be used with DiVE """
        df = self.data_frame
        keys = df.filename[:]

        values = df[df.columns[1:]]
        dict_of_dicts = {str(k): {"Coordinates": [float(x) for x in v]} for k,v in zip(keys, values.to_records(index=False))}
        return json.dumps(dict_of_dicts)



    def add_property(self, name, values):

        if len(values) != self.data_frame.size:
            raise Exception("Error number of values should be the same")

        self.property_names.append(name)
        self.properties.append(values)





    def __str__(self):
        return str(self.data_frame)


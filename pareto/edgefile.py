

import sys
import copy


class EdgeFile(object):
    """
    This class is a Python object representation of an edge file. The expected layout of the edge file is as follows:
    each line consists of three elements: two strings and a floating point number, separated by spaces
    """
    def __init__(self, filename, objScoreKey):

        # the file that the edge data is loaded from
        self.filename = filename

        # the 2-D, nominal parameter space
        self.parSpace = []

        # the objective score associated with a point in the parameter space
        self.objScores = []

        # create an empty set
        self.ulistphotos = set()

        # commence the loading from file
        if objScoreKey == 'goldberg':
            raise Exception('The key \'goldberg\' is reserved for calculating the Pareto scores later on.')
        else:
            self.load(objScoreKey)

    def load(self, objScoreKey):

        # read all the data from the edge file
        with open(self.filename, 'r') as f:
            rdata = f.read()

        lines = rdata.splitlines(False)
        for line in lines:
            photo1, photo2, objScore = line.split(' ')
            self.parSpace.append({'x': photo1, 'y': photo2})
            self.objScores.append({objScoreKey: objScore})

            if photo1 not in self.ulistphotos:
                self.ulistphotos.add(photo1)

            if photo2 not in self.ulistphotos:
                self.ulistphotos.add(photo2)


if __name__ == '__main__':

    obj1 = EdgeFile('../data/pentax/edgelist-pentax-ncc.txt', 'ncc')
    obj2 = EdgeFile('../data/pentax/edgelist-pentax-pce.txt', 'pce')

    print('Done.')



import sys

# Python 3 required
assert sys.version_info[0] == 3, "This code needs Python 3"


class EdgeFile(object):
    """
    This class is a Python object representation of an edge file. The expected layout of the edge file is as follows:
    each line consists of three elements: two strings and a floating point number, separated by spaces
    """
    def __init__(self, filename, objname, nrecordsmax=float('Inf'), lineartransform=None):

        # the file that the edge data is loaded from
        self.filename = filename

        # the 2-D, nominal parameter space
        self.parSpace = []

        # the objective score associated with a point in the parameter space
        self.objScores = []

        # create an empty set that will hold the set of photo file names
        self.usetphotos = set()

        # read all records from the edgefile, but store only this many as property of this instance
        self.nRecordsMax = nrecordsmax

        # initialize the linear transform to None by directly setting the 'private' property...
        self._lineartransform = None
        # ..but use the setter to really set its value
        if lineartransform is None:
            self.lineartransform = {'slope': 1, 'intercept': 0}
        else:
            self.lineartransform = lineartransform

        if objname == 'goldberg':
            raise Exception('The key \'goldberg\' is reserved for calculating the Pareto scores later on.')
        else:

            # the name of the objective (must be a valid python key)
            self.objName = objname

            # commence the loading from file
            self.load(objname)


    def __str__(self):
        """
        Override object's __str__ method with a pretty-print method of our own. This iterates over all elements in
        self.parSpace first, printing the x and y (there are always exactly two parSpace dimensions for this problem,
        because it's always a comparison between a pair of photographs), then adds the objective score
        :return: s -- string with pretty-printed representation of self.parSpace and self.objSpace
        """

        ncharsindent = 0
        for key in self.objScores[0].keys():
            if len(key) > ncharsindent:
                ncharsindent = len(key) + 2

        s = ''
        formatstr = '{: >' + str(ncharsindent) + '}: '
        for iElem in range(0, len(self.parSpace)):
            s += '{\n'
            s += formatstr.format('\'x\'')
            s += '\'' + str(self.parSpace[iElem]['x']) + '\''
            s += '\n'
            s += formatstr.format('\'y\'')
            s += '\'' + str(self.parSpace[iElem]['y']) + '\''
            s += '\n'

            s += formatstr.format('\'' + self.objName + '\'')
            s += str(self.objScores[iElem][self.objName])
            s += '\n'
            if iElem < len(self.parSpace) - 1:
                s += '},\n'
            else:
                s += '}\n'

        return s

    def print(self):
        # defer to self.__str__()
        print(self.__str__())

    def load(self, objname):

        # read all the data from the edge file
        with open(self.filename, 'r') as f:
            rdata = f.read()

        lines = rdata.splitlines(False)
        iline = 0
        for line in lines:
            photo1, photo2, objscore = line.split(' ')
            self.parSpace.append({'x': photo1, 'y': photo2})
            self.objScores.append({objname: float(objscore)})
            iline += 1

            if photo1 not in self.usetphotos:
                self.usetphotos.add(photo1)

            if photo2 not in self.usetphotos:
                self.usetphotos.add(photo2)

            if iline == self.nRecordsMax:
                print('Maximum number of records reached ({:d})'.format(self.nRecordsMax))
                break

    @property
    def lineartransform(self):
        return self._lineartransform

    @lineartransform.setter
    def lineartransform(self, lineartransform):

        if isinstance(lineartransform, dict):
            if 'slope' in lineartransform.keys() and 'intercept' in lineartransform.keys():
                self._lineartransform = lineartransform
            else:
                raise Exception('Input argument \'lineartransform\' does not have ' +
                                'the required keys [\'slope\',\'intercept\'].')
        else:
            raise Exception('Input argument \'lineartransform\' needs to be a dict.')

    def applytransform(self):

        for objscore in self.objScores:
            objscore[self.objName] = self.lineartransform['slope'] * objscore[self.objName] + \
                                     self.lineartransform['intercept']

if __name__ == '__main__':

    # obj1 = EdgeFile('../data/pentax/edgelist-pentax-ncc.txt', 'ncc', nrecordsmax=100)
    # obj2 = EdgeFile('../data/pentax/edgelist-pentax-pce.txt', 'pce')

    obj1 = EdgeFile('../data/paretotest/obj1.txt', 'obj1')
    obj1.print()

    print('\n')

    obj1.lineartransform = {'slope': 2, 'intercept': -1}
    obj1.applytransform()

    obj1.print()

    print('Done.')

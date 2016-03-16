#! /usr/bin/env python

# python3 style import:
from edgefile import EdgeFile

class EdgeFiles(object):
    """
    This class is used to merge different EdgeFiles. It keeps a master list of points in the parameter space for which
    objective scores have been calculated according to different metrics. Each element in self.parSpace i therefor
    associated with multiple scores in self.objSpace.
    """
    def __init__(self):
        """
        the constructor should be documented in this docstring
        """

        # the file that the edge data is loaded from
        self.filenames = []

        # the 2-D, nominal parameter space
        self.parSpace = None

        # the objective score associated with a point in the parameter space
        self.objSpace = None

        # create an empty set
        self.ulistphotos = None


    def __str__(self):

        nCharsIndent = 0
        for key in self.objSpace[0].keys():
            if len(key) > nCharsIndent:
                nCharsIndent = len(key) + 2

        s = ''
        formatStr = '{: >' + str(nCharsIndent) + '}: '
        for iElem in range(0, len(self.parSpace)):
            s += '{\n'
            s += formatStr.format('\'x\'')
            s += '\'' + str(self.parSpace[iElem]['x']) + '\''
            s += '\n'
            s += formatStr.format('\'y\'')
            s += '\'' + str(self.parSpace[iElem]['y']) + '\''
            s += '\n'
            for key in sorted(self.objSpace[iElem].keys()):
                s += formatStr.format('\'' + key + '\'')
                s += str(self.objSpace[iElem][key])
                s += '\n'
            if iElem < len(self.parSpace) - 1:
                s += '},\n'
            else:
                s += '}\n'

        return s


    def add(self, edgefile):

        """
        Add another EdgeFile and merge it with the existing objective score information on previous objectives, if any
        exist.
        :param edgefile:
        :return: self
        """

        isFirstTime = self.filenames == []

        self.filenames.append(edgefile.filename)

        if isFirstTime:

            self.parSpace = edgefile.parSpace
            self.objSpace = edgefile.objScores
            self.ulistphotos = edgefile.ulistphotos

        else:

            # check if ulistphotos has the same entries, raise exception otherwise
            if len(self.ulistphotos - edgefile.ulistphotos) == 0:
                print('Sets are equal. Proceeding with merge.')
            else:
                raise Exception('Sets from "' + self.filenames[0] + '" and "' +  edgefile.filename + '" are not equal. Aborting.')

            for iother in range(0, len(edgefile.parSpace)):
                iself = self.parSpace.index(edgefile.parSpace[iother])
                self.objSpace[iself].update(edgefile.objScores[iother])


    def calcPareto(self):
        """
        This method assumes that objective scores are positive numbers, and that the closer the score is to 0, the
        better it is.
        """

        def otherPointUnderPoint(op, p):

            if 'goldberg' in op.keys() and op['goldberg'] < iRank:
                return False

            opUnderP = True
            for key in p.keys():
                if op[key] >= p[key]:
                    opUnderP = False
                    break
            return opUnderP


        def hypercubeIsEmpty(point):

            # naively assume that hypercube is empty (initially)
            isempty = True
            for otherpoint in self.objSpace:
                if otherPointUnderPoint(otherpoint, point):
                    isempty = False
                    break
            return isempty


        # if the hypercube from the origin to the point contains no other points, the point is pareto-dominant
        nRanked = 0
        iRank = 0
        while nRanked < len(self.objSpace):
            iRank += 1
            for point in self.objSpace:
                if 'goldberg' in point.keys():
                    continue
                if hypercubeIsEmpty(point):
                    point.update({'goldberg': iRank})
                    nRanked += 1

        return self




if __name__ == '__main__':


    obj1 = EdgeFile('../data/paretotest/obj1.txt', 'obj1')
    obj2 = EdgeFile('../data/paretotest/obj2.txt', 'obj2')

    edgeFiles = EdgeFiles()
    edgeFiles.add(obj1)
    edgeFiles.add(obj2)

    edgeFiles.calcPareto()

    print(edgeFiles)

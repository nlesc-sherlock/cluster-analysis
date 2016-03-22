#! /usr/bin/env python

import sys
from edgefile import EdgeFile
import plotly.offline as offlineplotly
import plotly.graph_objs as grob
from plotly import tools

# Python 3 required
assert sys.version_info[0] == 3, "This code needs Python 3"


class EdgeFiles(object):
    """
    This class is used to merge different EdgeFiles. It keeps a master list of points in the parameter space for which
    objective scores have been calculated according to different metrics. Each element in self.parSpace is therefore
    associated with multiple scores in self.objSpace.
    """
    def __init__(self, assumesameorder=True, filename='temp-plot.html'):
        """
        the constructor should be documented in this docstring
        """

        # the list of files that the edge data is loaded from
        self.filenames = None

        # the 2-D, nominal parameter space
        self.parSpace = None

        # the objective score(s) associated with a point in the parameter space
        self.objSpace = None

        # the name(s) of the objective functions
        self.objNames = None

        # create an empty set that will hold a list of file names of the photos
        self.usetphotos = None

        # assume that any EdgeFile object is ordered exactly equally
        self.assumeSameOrder = assumesameorder

        # define the filename of the plotly figure
        self.filename = filename

    def __str__(self):
        """
        Override object's __str__ method with a pretty-print method of our own. This iterates over all elements in
        self.parSpace first, printing the x and y (there are always exactly two parSpace dimensions for this problem,
        because it's always a comparison between a pair of photographs), then adds the objective scores as well and
        finally also prints the pareto score if it exists
        :return: s -- string with pretty-printed representation of self.parSpace and self.objSpace
        """

        ncharsindent = 0
        for key in self.objSpace[0].keys():
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
            for key in sorted(self.objSpace[iElem].keys()):
                s += formatstr.format('\'' + key + '\'')
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

        if self.filenames is None:

            self.filenames = [edgefile.filename]

            self.parSpace = edgefile.parSpace

            self.objSpace = edgefile.objScores

            self.objNames = [edgefile.objName]

            self.usetphotos = edgefile.usetphotos

        else:

            # check if usetphotos has the same entries, raise exception otherwise
            if len(self.usetphotos - edgefile.usetphotos) == 0:
                print('Sets are equal. Proceeding with merge.')
            else:
                raise Exception('Sets from "' + self.filenames[0] + '" and "' + edgefile.filename +
                                '" are not equal. Aborting.')

            self.filenames.append(edgefile.filename)

            self.objNames.append(edgefile.objName)

            if self.assumeSameOrder:
                for iother in range(0, len(edgefile.parSpace)):
                    iself = iother
                    self.objSpace[iself].update(edgefile.objScores[iother])
            else:
                for iother in range(0, len(edgefile.parSpace)):
                    try:
                        iself = self.parSpace.index(edgefile.parSpace[iother])
                    except ValueError:
                        print(str(edgefile.parSpace[iother]) + ' does not occur in self.parSpace.')
                        continue

                    self.objSpace[iself].update(edgefile.objScores[iother])

    def calcpareto(self):
        """
        This method assumes that objective scores are positive numbers, and that the closer the score is to 0, the
        better it is.
        """

        def otherpointunderpoint(op, p):
            """
            Calculate whether op is under p, i.e. whether op is smaller than p in all dimensions
            :param op: other point, n-dimensional dict with a numerical score for each objective
            :param p: point, n-dimensional dict with a numerical score for each objective
            :return: True if there is another point under point, i.e. smaller in all dimensions, False otherwise
            """

            if 'goldberg' in op.keys() and op['goldberg'] < irank:
                return False

            opunderp = True
            for key in p.keys():
                if op[key] >= p[key]:
                    opunderp = False
                    break
            return opunderp

        def hyperrectisempty():
            """
            Determine whether there is any point in self.objSpace which is under point, and if so return False because
            in that case hyperrect is not empty
            :return:
            """

            # initially assume that hyperrect is empty
            isempty = True
            for otherpoint in self.objSpace:
                if otherpointunderpoint(otherpoint, point):
                    isempty = False
                    break
            return isempty

        # if the hyperrect from the origin to the point contains no other points, the point is pareto-dominant
        nranked = 0
        irank = 0
        while nranked < len(self.objSpace):
            irank += 1
            for point in self.objSpace:
                if 'goldberg' in point.keys():
                    continue
                if hyperrectisempty():
                    point.update({'goldberg': irank})
                    nranked += 1

        return self

    def show(self, dimensions=None):

        def getscatterobj(xobjname, yobjname):
            """
            Constructs a 2-D Scatter trace object where each point is a combination of the objective score for objective
            'xobjname' and of the objective score of objective 'yobjname'.
            :param xobjname: objective score for the horizontal axis
            :param yobjname: objective score for the vertical axis
            :return: plotly Scatter object
            """

            x = []
            y = []
            ranktext = []
            for item in self.objSpace:
                x.append(item[xobjname])
                y.append(item[yobjname])
                ranktext.append('rank: ' + str(item['goldberg']))

            return grob.Scatter(
                x=x,
                y=y,
                text=ranktext,
                marker=grob.Marker(
                    symbol='cross',
                    color='rgb(237,0,178)',
                    size=6
                ),
                mode='markers',
                name='{:s} v {:s}'.format(xobjname, yobjname),
                opacity=0.7
            )

        def getaxes(idim, direction):
            """
            Returns a plotly xaxis or a plotly yaxis, with properties. They are stored together as a dictionary.
            :param idim: this parameter indicates which dimension out of 'dimensions' we currently need an axis for
            :param direction: this parameter indicates whether we are looking for an xaxis or a yaxis.
            :return: a dictionary containing either (1) the dictionary key 'xaxisN' where N is an integer, followed by
            its value, a plotly XAxis with properties; or (2) the dictionary key 'yaxisN' where N is an integer,
            followed by its value, a plotly YAxis with properties
            """

            if direction == 'x':
                axisstr = direction + 'axis' + str(idim + 1)
                return {axisstr: grob.XAxis(
                        autorange=True,
                        showgrid=True,
                        showline=True,
                        title=dimensions[idim],
                        zeroline=False,
                        type=u'linear',
                        ticks=u'outside',
                        mirror='allticks',
                        linecolor=u'rgb(16, 16, 16)',
                        gridcolor=u'rgb(185, 185, 185)',
                        linewidth=1)}
            elif direction == 'y':
                axisstr = direction + 'axis' + str(ndims - idim)
                return {axisstr: grob.YAxis(
                        autorange=True,
                        showgrid=True,
                        showline=True,
                        title=dimensions[idim],
                        zeroline=False,
                        type=u'linear',
                        ticks=u'outside',
                        mirror='allticks',
                        linecolor=u'rgb(16, 16, 16)',
                        gridcolor=u'rgb(185, 185, 185)',
                        linewidth=1)}
            else:
                raise Exception('Your axis should be \'x\' or \'y\'.')

        # check the input type
        if not isinstance(dimensions, list):
            raise Exception('"Input argument \'dimensions\' should be a list, but you\'ve provided a ' +
                            str(type(dimensions)) + '"')

        if dimensions is None:
            dimensions = self.objNames

        for dim in dimensions:
            if dim not in self.objNames:
                raise Exception('"You want to plot a dimension that doesn\'t exist."')

        # how many dimensions we're plotting
        ndims = len(dimensions)

        # don't open new browser tabs/windows every time you run the script:
        auto_open = False

        # define the number of rows and columns of subplots, say how the axes are linked together, and how much space
        # there needs to be in between axes
        fig = tools.make_subplots(rows=ndims,
                                  cols=ndims,
                                  shared_xaxes=True,
                                  shared_yaxes=True,
                                  vertical_spacing=0.025,
                                  horizontal_spacing=0.025)

        # define the background color of the figure
        fig['layout'].update(paper_bgcolor='rgba(255,255,255,255)')

        # define the background color of the axes
        fig['layout'].update(plot_bgcolor='rgba(222,222,222,255)')

        for irow in range(0, ndims):
            for icol in range(0, ndims):

                # define a data series (or trace in plotly speak)
                trace = getscatterobj(dimensions[icol], dimensions[irow])

                # append the trace to the list of traces
                fig.append_trace(trace, ndims - irow, icol + 1)

                # define which axes the trace should be plotted against
                fig['layout'].update(getaxes(icol, 'x'))
                fig['layout'].update(getaxes(irow, 'y'))

        # don't show the legend
        fig['layout'].update(showlegend=False)

        # start drawing with the current settings
        offlineplotly.plot(fig, auto_open=auto_open, filename=self.filename)

        return None

    def print(self):
        # defer to self.__str__()
        print(self.__str__())


if __name__ == '__main__':

    # make a python object representation of the data in these two files:
    obj1 = EdgeFile('../data/paretotest/obj1.txt', 'obj1')
    obj2 = EdgeFile('../data/paretotest/obj2.txt', 'obj2')

    # initialize the object that will merge all the info from separate EdgeFile objects
    edgeFiles = EdgeFiles()

    # add the EdgeFile objects, merging with any previous objects
    edgeFiles.add(obj1)
    edgeFiles.add(obj2)

    # calc the goldberg pareto scores given the objective scores we just added
    edgeFiles.calcpareto()

    # plot the objective space
    edgeFiles.show(['obj1', 'obj2'])

    edgeFiles.print()

    # # make a python object representation of the data in these two files:
    # ncc = EdgeFile('../data/pentax/edgelist-pentax-ncc.txt', 'ncc')
    # pce = EdgeFile('../data/pentax/edgelist-pentax-pce.txt', 'pce')
    # pce0 = EdgeFile('../data/pentax/edgelist-pentax-pce0.txt', 'pce0')
    #
    # # initialize the object that will merge all the info from separate EdgeFile objects
    # edgeFiles = EdgeFiles()
    #
    # # add the EdgeFile objects, merging with any previous objects
    # edgeFiles.add(ncc)
    # edgeFiles.add(pce)
    # edgeFiles.add(pce0)
    #
    # # calc the goldberg pareto scores given the objective score we just added
    # edgeFiles.calcpareto()
    #
    # # plot the objective space
    # edgeFiles.show(['ncc', 'pce', 'pce0'])
    #
    # # pretty-print the result
    # print(edgeFiles)

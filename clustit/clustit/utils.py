""" util functions for use in clustit """
import numpy

class MatrixReadError(Exception):
    pass

class UnknownFileFormat(Exception):
    pass

def read_edgelist_file(filename):
    """ reads an edgelist file, returns a structured array """
    edgelist = numpy.genfromtxt(filename, dtype=None, names=['n1','n2','d'], delimiter=" ")
    return edgelist

def read_distance_matrix_file(filename):
    """ reads a distance matrix from file stored as either a text or binary file """
    extension = filename.split(".")[-1]
    if extension == "dat":
        #try both little and big endian and chose the format
        #that produces the smallest exponents, a little dirty but effective
        matrix = numpy.fromfile(filename, dtype='d')
        explen_l = len(matrix.max().__format__('e').split('e')[-1])
        matrix = numpy.fromfile(filename, dtype='>d')
        explen_b = len(matrix.max().__format__('e').split('e')[-1])
        if explen_b > explen_l:
            matrix = numpy.fromfile(filename, dtype='d')
        numrows = int(numpy.sqrt(matrix.size))
        if numrows * numrows != matrix.size:
            raise MatrixReadError("Expected square matrix of 64bit floating point numbers")
        matrix = matrix.reshape(numrows, numrows)
    elif extension == "txt":
        matrix = numpy.genfromtxt(filename, delimiter=",")
        if len(matrix.shape) != 2:
            raise MatrixReadError("Expected square 2D comma-separated matrix, one row per line")
        n = min(matrix.shape)
        matrix = matrix[0:n,0:n]
    else:
        raise UnknownFileFormat("Expected .dat or .txt file")
    return matrix

def edgelist_to_distance_matrix(edgelist):
    """ creates a distance matrix out of an edgelist, also returns list of names """
    #get unique names in n1, while preserving the order in n1
    #(using set() for n1 is not an option)
    names1 = set(edgelist['n1'])
    n1 = []
    for i in edgelist['n1']:
        if i in names1 and i not in n1:
            n1.append(i)
            names1.discard(i)
    n2 = set([n for n in edgelist['n2'] if n not in n1])
    #could insert sanity check here that n2 has length exactly 1
    names = list(n2) + n1
    n = len(names)
    matrix = numpy.zeros((n,n), dtype=numpy.float)
    for i,d in enumerate(edgelist['d']):
        n1 = edgelist[i]['n1']
        n2 = edgelist[i]['n2']
        mi = names.index(n1)
        mj = names.index(n2)
        matrix[mi,mj] = d
        matrix[mj,mi] = d
    numpy.fill_diagonal(matrix, 0.0)
    return matrix, names

def distance_matrix_to_edgelist(matrix, names=None):
    """ creates an edgelist out of a distance matrix, using names if provided """
    names = names or ['node' + str(i) for i in range(matrix.shape[0])]
    max_len = len(max(names, key=len))

    edgelist = []
    for j in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            if i<j:
                edgelist.append((names[j], names[i], matrix[j,i]))
    edgelist = numpy.array(edgelist, dtype=[('n1', 'S'+str(max_len)),
        ('n2', 'S'+str(max_len)), ('d', 'f64')])
    return edgelist

def similarity_to_distance(similarities, cutoff):
    """ convert the edgelist and/or matrix from similarity to distance using cutoff """
    distances = similarities
    distances[distances > cutoff] = cutoff
    distances[distances < 0.0] = 0.0
    distances += 0.0000001
    distances = cutoff / distances
    distances[distances > cutoff] = cutoff
    if len(distances.shape) == 2:
        numpy.fill_diagonal(distances, 0.0)
    return distances

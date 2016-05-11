from .context import clustit
import numpy
from nose.tools import raises
import clustit.utils as utils

import os
test_files = os.path.dirname(os.path.realpath(__file__)) + "/test_files/"

def test_read_edgelist_file():
    output = utils.read_edgelist_file(test_files + 'edgelist.txt')
    assert output.size == 3
    assert output['n1'][0] == 'node002'
    assert output['n2'][0] == 'node001'
    assert output['d'][0] == 5.7

def test_read_distance_matrix_file1():
    output = utils.read_distance_matrix_file(test_files + 'matrix_big_endian.dat')
    assert output.size == 9
    assert output.shape == (3,3)
    assert output[0][0] == 0.0
    assert output[0][1] == 1.0
    assert output[0][2] == 2.0

def test_read_distance_matrix_file2():
    output = utils.read_distance_matrix_file(test_files + 'matrix_little_endian.dat')
    assert output.size == 9
    assert output.shape == (3,3)
    assert output[0][0] == 0.0
    assert output[0][1] == 1.0
    assert output[0][2] == 2.0

@raises(utils.UnknownFileFormat)
def test_read_distance_matrix_file3():
    output = utils.read_distance_matrix_file(test_files + 'bogus_file.file')

def test_edgelist_to_distance_matrix():
    edgelist = numpy.array([('node002', 'node001', 2.0), ('node003', 'node001', 4.0),
                    ('node003', 'node002', 1.2)], dtype=[('n1', 'S7'),('n2', 'S7'),('d', 'f64')])

    matrix, names = utils.edgelist_to_distance_matrix(edgelist)
    expected = ['node001', 'node002', 'node003']
    assert all([n == e for n,e in zip(names,expected)])
    assert matrix.shape == (3,3)
    assert matrix[0][0] == 0.0
    assert matrix[1][0] == 2.0
    assert matrix[2][0] == 4.0

def test_distance_matrix_to_edgelist():
    matrix = numpy.array([[0, 1, 2],[1,0,3],[2,3,0]])
    output = utils.distance_matrix_to_edgelist(matrix)
    assert len(output) == 3
    expected_n1 = ['node1', 'node2', 'node2']
    expected_n2 = ['node0', 'node0', 'node1']
    expected_d  = [1, 2, 3]
    assert all([n == e for n,e in zip(output['n1'],expected_n1)])
    assert all([n == e for n,e in zip(output['n2'],expected_n2)])
    assert all([n == e for n,e in zip(output['d'],expected_d)])


from __future__ import print_function

from .context import clustit
import numpy
import clustit.embedding as embedding
from clustit.output import OutputCollection

import json

import os
test_files = os.path.dirname(os.path.realpath(__file__)) + "/test_files/"


def test_output():
    df = embedding.largevis(test_files + 'edgelist-pentax-pce.txt', outdim=3, alpha=0.1)

    output_collection = OutputCollection(df)

    print(output_collection)

    json_string = output_collection.to_DiVE()
    print(json_string)

    try:
        json_object = json.loads(json_string)
    except ValueError as e:
        assert False

    assert True


def test_output2():
    df = embedding.largevis(test_files + 'edgelist-pentax-pce.txt', outdim=3, alpha=0.1)
    oc = OutputCollection(df)

    array = oc.to_array()
    assert array.shape == (5,3)


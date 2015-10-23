#!/usr/bin/env python

import os
import sqlite3 as sqlt
import argparse as argp


parser = argp.ArgumentParser(
   description = 'This script generates a sample of N sub-graphs from an input graph stored in a database.')

parser.add_argument(
   '-f',
   required = True,
   type = float,
   dest = 'fraction',
   help = 'select a sub-set of edges from graph G (fraction)')

parser.add_argument(
   '-n',
   required = True,
   type = int,
   dest = 'n_graphs',
   help = 'generate a sample of N graphs')

parser.add_argument(
   '-c',
   required = False,
   type = float,
   dest = 'cutoff',
   default = 0,
   help = 'edge weight (similarity) cutoff')

parser.add_argument(
   'dbfile',
   help = 'SQLite database with graph G')

args = parser.parse_args()
fraction = float(args.fraction)
n_graphs = int(args.n_graphs)
cutoff = float(args.cutoff)
dbfile = args.dbfile
sep = '\t'

if n_graphs < 1 or fraction <= 0 or fraction >= 1:
   parser.print_help()

if os.path.isfile(dbfile) is False:
      parser.error("dbfile '%s' not found" % dbfile)

with sqlt.connect(dbfile) as conn:
   stmt_1 = """
      SELECT COUNT(*)
      FROM GRAPH
      WHERE weight > %f
   """ % cutoff

   cur = conn.cursor()
   cur.execute(stmt_1)
   n_edges = cur.fetchone()[0]

   stmt_2 = """
      SELECT *
      FROM GRAPH
      WHERE weight > %f
      ORDER by RANDOM()
      LIMIT %d
   """ % (cutoff, fraction * n_edges)
   
   for i in range(1, n_graphs + 1):
      outfile = '{basename}.n{n_graphs}_f{fraction}_c{cutoff}.graph.{i}'.format(
         basename = os.path.splitext(os.path.basename(dbfile))[0],
         n_graphs = n_graphs,
         fraction = fraction,
         cutoff = cutoff,
         i = i)

      with open(outfile, 'w+') as fout:
         for row in cur.execute(stmt_2):
            fout.write(sep.join([ str(x) for x in row ]) + os.linesep)

   cur.close()

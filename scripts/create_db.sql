CREATE TABLE GRAPH (
   node1 TEXT,
   node2 TEXT,
   weight NUMERIC,
   PRIMARY KEY(node1, node2)
);

--.sep \t
--.import FILENAME GRAPH

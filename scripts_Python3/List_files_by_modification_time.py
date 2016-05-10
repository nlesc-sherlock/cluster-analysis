# #!/usr/bin/env python
# from stat import S_ISREG, ST_CTIME, ST_MODE
# import os, sys, time
# 
# # path to the directory (relative or absolute)
# dirpath = sys.argv[1] if len(sys.argv) == 2 else r'.'
# dirpath = '/home/hspreeuw'
# # get all entries in the directory w/ stats
# entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
# entries = ((os.stat(path), path) for path in entries)
# 
# # leave only regular files, insert creation date
# entries = ((stat[ST_CTIME], path)
#            for stat, path in entries if S_ISREG(stat[ST_MODE]))
# #NOTE: on Windows `ST_CTIME` is a creation date 
# #  but on Unix it could be something else
# #NOTE: use `ST_MTIME` to sort by a modification date
# 
# for cdate, path in sorted(entries):
#     print(time.ctime(cdate), os.path.basename(path))

import optparse
import os
import fnmatch
import time

# Parse options
parser = optparse.OptionParser(usage='Usage: %prog [options] path [path2 ...]')
parser.add_option('-g', action='store', type='long', dest='secs', default=10,
                  help='set threshold for grouping files')
parser.add_option('-f', action='append', type='string', dest='exc_files', default=[],
                  help='exclude files matching a wildcard pattern')
parser.add_option('-d', action='append', type='string', dest='exc_dirs', default=[],
                  help='exclude directories matching a wildcard pattern')
options, roots = parser.parse_args()

if len(roots) == 0:
    print('You must specify at least one path.\n')
    parser.print_help()

def iterFiles(options, roots):
    """ A generator to enumerate the contents directories recursively. """
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            name = os.path.split(dirpath)[1]
            if any(fnmatch.fnmatch(name, w) for w in options.exc_dirs):
                del dirnames[:]  # Don't recurse here
                continue
            stat = os.stat(os.path.normpath(dirpath))
            yield stat.st_mtime, '', dirpath  # Yield directory
            for fn in filenames:
                if any(fnmatch.fnmatch(fn, w) for w in options.exc_files):
                    continue
                path = os.path.join(dirpath, fn)
                stat = os.lstat(os.path.normpath(path))  # lstat fails on some files without normpath
                mtime = max(stat.st_mtime, stat.st_ctime)
                yield mtime, stat.st_size, path  # Yield file

# Build file list, sort it and dump output
ptime = 0
for mtime, size, path in sorted(iterFiles(options, roots), reverse=True):
    if ptime - mtime >= options.secs:
        print('-' * 30)
    timeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    print('%s %10s %s' % (timeStr, size, path))
    ptime = mtime
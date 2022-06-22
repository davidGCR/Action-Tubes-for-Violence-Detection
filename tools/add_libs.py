import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = os.path.dirname(os.path.dirname(__file__))
# lib_path = os.path.join(vionet_dir, 'lib')


# tube_dir = os.path.join(os.path.dirname(vionet_dir),'TubeletGeneration')
# customdataset_dir = os.path.join(os.path.dirname(vionet_dir),'customsdatasets')
# transformation_dir = os.path.join(os.path.dirname(vionet_dir),'transformations')


libraries = [root_dir]#[src_path, vionet_dir, lib_path, tube_dir, customdataset_dir, transformation_dir]

# add_path(os.path.normpath(lib_path))
# add_path(os.path.normpath(tube_dir))

for l in libraries:
    print('adding lib: ', l)
    add_path(os.path.normpath(l))
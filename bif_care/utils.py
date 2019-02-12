import os
import re
import pathlib
import javabridge as jv
import bioformats as bf
from collections import namedtuple

Axes = namedtuple("Axes", "t z c y x")

class JVM(object):
    log_config = "res/log4j.properties"
    started = False

    def start(self):
        if not JVM.started:
            jv.start_vm(class_path=bf.JARS, 
                        max_heap_size='8G', 
                        args=["-Dlog4j.configuration=file:{}".format(self.log_config),],
                        run_headless=True)
            JVM.started = True

    def shutdown(self):
        if JVM.started:
            jv.kill_vm()
            JVM.started = False

def get_pixel_dimensions(fn):
    JVM().start()

    
    ir = bf.ImageReader(str(fn))
    
    t_size = ir.rdr.getSizeT()
    z_size = ir.rdr.getSizeZ()
    c_size = ir.rdr.getSizeC()
    y_size = ir.rdr.getSizeY()
    x_size = ir.rdr.getSizeX()
    
    ir.close()
    return Axes(t_size, z_size, c_size, y_size, x_size)

def get_file_list(in_dir, glob):
    assert os.path.exists(in_dir), "Folder '{}' does not exist".format(in_dir)
    return sorted(list( pathlib.Path(in_dir).glob(glob)))

def check_file_lists(in_dir, low_wc, high_wc):
    from fnmatch import translate
    fl_low = get_file_list(in_dir, low_wc)
    fl_high = get_file_list(in_dir, high_wc)

    if len(fl_low) == 0:
        return False, "No files selected"

    if len(fl_low) != len(fl_high):
        return False, "Number of files does not match {} != {}".format(len(fl_low), len(fl_high))

    for fl, fh in zip(fl_low, fl_high):
        if os.path.splitext(fl.name)[1] != os.path.splitext(fh.name)[1]:
            return False, "Extensions do not match"

        dim_low  = get_pixel_dimensions(fl)
        dim_high = get_pixel_dimensions(fh)

        if dim_low.c != dim_high.c:
            return False, "Low and high quality images have different channels\n '{}' != '{}'".format(fl, fh)

        if dim_low.t != dim_high.t:
            return False, "Low and high quality images have different number of time points\n '{}' != '{}'".format(fl, fh)
        
        if (dim_low.x > dim_high.x) or \
           (dim_low.y > dim_high.y) or \
           (dim_low.z > dim_high.z):
           return False, "Low quality images have higher spatial resolution"

        if (dim_low.t > dim_low.z) and \
            (dim_low.z == 1):
            return False, "Only 1 z-slice in '{}' but {} frames. Make sure input images are Z-stacks.".format(fl, dim_low.t)

    return True, "OK"




    





def get_upscale_factors(in_dir, low_wc, high_wc): 
    low_fl = get_file_list(in_dir, low_wc)
    high_fl = get_file_list(in_dir, high_wc)        

    low_dim  = get_pixel_dimensions(str(low_fl[0]))
    high_dim = get_pixel_dimensions(str(high_fl[0]))

    return (high_dim.z / low_dim.z,
            high_dim.y / low_dim.y,
            high_dim.x / low_dim.x)


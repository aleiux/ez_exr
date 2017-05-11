import ez_exr
from tools import *
import re
from os import listdir
from os.path import isfile, join

def load_frame(im_name, z_name):
	"""
	Returns a frame
	"""
	new_frame = Frame(ez_exr.read_image(im_name), ez_exr.read_image(z_name))
	return new_frame

if __name__ == "__main__":
	background = load_frame("composite_test/background.exr", "composite_test/background_z.exr")
	subject = load_frame("composite_test/subject.exr", "composite_test/subject_z.exr")
	holdout = load_frame("composite_test/holdout.exr", "composite_test/holdout_z.exr")

	result = z_comp(subject, background)
	ez_exr.write_image("result.exr", result.image)
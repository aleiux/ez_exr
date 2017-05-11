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
	if False:
		background = load_frame("composite_test_stuff/background.exr", "composite_test_stuff/background_z.exr")
		subject = load_frame("composite_test_stuff/subject.exr", "composite_test_stuff/subject_z.exr")
		#holdout = load_frame("composite_test/holdout.exr", "composite_test/holdout_z.exr")
		result = z_comp(subject, background)
		ez_exr.write_image("result.exr", result.image)
	if False:
		to_reduce = load_frame("anim_shadows/composite_test_filtered.0001.exr", "anim_shadows/composite_test_z.0001.exr")
		result = shadow_denoise(to_reduce)
		ez_exr.write_image("result.exr", result.image)
	if False:
		for i in range(1, 25):
			to_reduce = load_frame("anim_shadows/composite_test_filtered.{}.exr".format(str(i).zfill(4)), "anim_shadows/composite_test_z.{}.exr".format(str(i).zfill(4)))
			result = shadow_denoise(to_reduce)
			ez_exr.write_image("shadows_result/result.{}.exr".format(str(i).zfill(4)), result.image)
	if True:
		for i in range(1, 25):
			background = load_frame("composite_test/background.exr", "composite_test/background_z.exr")
			shadow = load_frame("anim_shadows/composite_test_filtered.{}.exr".format(str(i).zfill(4)),  "anim_shadows/composite_test_z.{}.exr".format(str(i).zfill(4)))
			shadow = shadow_denoise(shadow)
			shadow.z[:, :] = background.z[:, :] + 1e-7
			subject =  load_frame("source_composite_subject/composite_test_filtered.{}.exr".format(str(i).zfill(4)), "source_composite_subject/composite_test_z.{}.exr".format(str(i).zfill(4)))
			shadow.image[:, :, 3] = np.clip(shadow.image[:, :, 3] * 1.6, 0, 1)
			result = z_comp(background, shadow)
			result = z_comp(result, subject)
			ez_exr.write_image("final_anim/result.{}.exr".format(str(i).zfill(4)), result.image)
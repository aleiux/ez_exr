import numpy as np
from numpy import inf
from scipy.ndimage import convolve
import itertools

class Frame:
	"""
	Holds related buffers for a single frame. Cleans up z infinity values to be 0. Also corrects for fringing artefacts
	"""
	def __init__(self, image, z = None):
		self.image = image
		if z is not None:
			z[z == -inf] = 0.
			z[z == inf] = 0.
			z  = z * image[:, :, 3:4]
		self.z = z
def z_comp(frame0, frame1):
	"""
	Performs Z buffer compositing on the two images. Returns new frame
	Assumes RGBA. Also assumes the Z buffer is "normalized" to 1 / depth (so closer objects have larger values)
	"""
	c0 = frame0.image[:, :, :3] * frame0.image[:, :, 3:4]
	c1 = frame1.image[:, :, :3] * frame1.image[:, :, 3:4]
	
	top_filter = (frame1.z > frame0.z).astype(float)
	edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).astype(float)
	edges = convolve(top_filter[:, :, 0], edge_filter, mode = 'reflect')
	#perform aliasing cleanup
	zs0 = frame0.z.astype(float)[:, :, 0]
	zs1 = frame1.z.astype(float)[:, :, 0]
	zs0[frame0.image[:, :, 3] != 1.0] = 0.
	zs1[frame1.image[:, :, 3] != 1.0] = 0.
	zs0[edges != 0] = 0.
	zs1[edges != 0] = 0.
	zs0 = np.pad(zs0, 1, 'constant')
	zs1 = np.pad(zs1, 1, 'constant')
	
	#return Frame(np.dstack((edges < 0, np.zeros(edges.shape), edges > 0)))
	
	for i, j in itertools.product(range(edges.shape[0]), range(edges.shape[1])):
		if edges[i][j] > 0:
			if frame1.image[i][j][3] == 1. and frame0.image[i][j][3] > 0:
				window = zs1[i-1:i+2, j-1:j+2]
				#window = np.abs(window - window[1][1])
				#window[1][1] = inf
				mi, mj = np.unravel_index(window.argmax(), window.shape)
				c1[i][j] = c1[i + mi - 1][j + mj -1] 
		elif edges[i][j] < 0:
			if frame0.image[i][j][3] == 1. and frame1.image[i][j][3] > 0:
				window = zs0[i-1:i+2, j-1:j+2]
				#window = np.abs(window - window[1][1])
				#window[1][1] = inf
				mi, mj = np.unravel_index(window.argmax(), window.shape)
				c0[i][j] = c0[i + mi - 1][j + mj -1]
	
	im_result = np.zeros(frame0.image.shape, frame0.image.dtype)
	im_result[:, :, :3] = c0 - (c0 * frame1.image[:, :, 3:4] * (top_filter)) + c1 - (c1 * frame0.image[:, :, 3:4] * (1-top_filter))
	im_result[:, :, 3] = frame0.image[:, :, 3] + frame1.image[:, :, 3] * (1 - frame0.image[:, :, 3])
	#im_result[np.isclose(im_result[:, :, 3], 1.0), 3] == 1.0
	z_result = np.maximum(frame0.z, frame1.z)
	return Frame(im_result, z_result)
	
	"""get top_filter
		wherever top filter is 0, frame0 is in front. wherever top filter is 1, frame1 is in front
		wherever edges is negative, frame 0 is in front 
		wherever edges is positive, frame 1 is in front
		if pixel has full alpha:
		for each pixel where edge is negative, search in frame0 neighborhood of pixel to find the closest z value where alpha = 1
		replace color with this, but keep the original alpha
		for each pixel where edge is positive, search in frame1 neighborhood and do the same
		if pixel has partial alpha, then AA is already taken care of
		also only do it if pixel is in front of the other image. that is other image alpha is > 0
	"""
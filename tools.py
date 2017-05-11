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
	zs0[frame0.image[:, :, 3] != 1.0] = 0. #set search z to 0 if semitransparent
	zs1[frame1.image[:, :, 3] != 1.0] = 0.
	zs0[edges != 0] *= 0.
	zs1[edges != 0] *= 0.
	zs0 = np.pad(zs0, 1, 'constant')
	zs1 = np.pad(zs1, 1, 'constant')
	zr0 = np.pad(frame0.z.astype(float)[:, :, 0], 1, 'constant')
	zr1 = np.pad(frame1.z.astype(float)[:, :, 0], 1, 'constant')
	for i, j in itertools.product(range(edges.shape[0]), range(edges.shape[1])):
		if edges[i][j] > 0:
			if frame1.image[i][j][3] == 1. and frame0.image[i][j][3] > 0 and frame1.z[i][j] != 0:
				window = zs1[i:i+3, j:j+3]
				window_norm = np.abs(window * (1. / frame1.z[i][j]) - 1.0) #normalize
				mi, mj = np.unravel_index(window_norm.argmin(), window_norm.shape)
				c1[i][j] = c1[abs(i + mi - 1)][abs(j + mj -1)]
				z_here = frame1.z[i][j]

				window_0 = zr0[i:i+3, j:j+3]
				window_1 = zr1[i:i+3, j:j+3]
				sandwiches = np.logical_and(z_here > window_0,  window_0 > window_1).astype(float)
				frame1.image[i][j][3] = 1 - np.sum(sandwiches) / 8.
		elif edges[i][j] < 0:
			if frame0.image[i][j][3] == 1. and frame1.image[i][j][3] > 0 and frame0.z[i][j] != 0:
				window = zs0[i:i+3, j:j+3]
				window = np.abs(window * (1. / frame0.z[i][j]) - 1.0)
				mi, mj = np.unravel_index(window.argmin(), window.shape)
				c0[i][j] = c0[i + mi - 1][j + mj -1]
				z_here = frame0.z[i][j]

				window_0 = zr0[i:i+3, j:j+3]
				window_1 = zr1[i:i+3, j:j+3]
				sandwiches = np.logical_and(z_here > window_1,  window_1 > window_0).astype(float)
				frame0.image[i][j][3] = 1 - np.sum(sandwiches) / 8.
				
				
	im_result = np.zeros(frame0.image.shape, frame0.image.dtype)
	im_result[:, :, :3] = c0 - (c0 * frame1.image[:, :, 3:4] * (top_filter)) + c1 - (c1 * frame0.image[:, :, 3:4] * (1-top_filter))
	im_result[:, :, 3] = frame0.image[:, :, 3] + frame1.image[:, :, 3] * (1 - frame0.image[:, :, 3])
	im_result[np.isclose(im_result[:, :, 3], 1.0), 3] == 1.0
	z_result = np.maximum(frame0.z, frame1.z)
	return Frame(im_result, z_result)
	
	"""get top_filter
		wherever top filter is 0, frame0 is in front. wherever top filter is 1, frame1 is in front
		wherever edges is negative, frame 0 is in front 
		wherever edges is positive, frame 1 is in front
		if pixel has full alpha: (if pixel has partial alpha, then AA is already taken care of)
		for each pixel where edge is negative, search in frame0 neighborhood of pixel to find the closest z value where alpha = 1
		replace color with this, but keep the original alpha. Only search where pixels are not on edge
		for each pixel where edge is positive, search in frame1 neighborhood and do the same
		
		also only do it if pixel is in front of the other image. that is other image alpha is > 0
		----
		Approximating alpha:
		For any particular pixel p on layer L, if p's Z value is greater than another pixel q in a different layer K, then 
		we may need to do alpha blending. This should occur when q in L has a lesser z value than q in K.
		The number of such pixels determines the alpha, where 8 is the maximum (alpha = 0
		i.e. if L_p > K_q > L_q for all q in window, for some p (middle)
		
	"""
import array
import OpenEXR
import Imath
import numpy as np

def autodetect_channels(header):
	"""
	Attempts to detect what channels are available and returns, in order of priority:
	RGBA, RGB, Z 
	Will detect if rgb(a) should be in upper or lower case. Assumes z is lower case.
	"""
	avail = header["channels"]
	channels = None
	uppercase = "A" in avail
	has_alpha = "A" in avail or "a" in avail
	has_rgb = ("R" in avail or "r" in avail) and ("G" in avail or "g" in avail) and ("B" in avail or "b" in avail)
	has_z = "Z" in avail or "z" in avail
	has_z_as_R = "R" in avail and "G" not in avail
	if has_rgb: 
		if has_alpha:
			if uppercase:
				channels = ("R", "G", "B", "A")
			else:
				channels = ("r", "g", "b", "a")
		else:
			if uppercase:
				channels = ("R", "G", "B")
			else:
				channels = ("r", "g", "b")
	else:
		if has_z: 
			channels = ("z")
		elif has_z_as_R:
			channels = ("R")
		else: 
			channels = None
	return channels
	
def read_image(file, channels = None):
	"""
	If file is a string, read_image will open it as a filename. 
	Otherwise it will treat it as an OpenEXR.InputFile.
	if channels is not provided, read_image will attempt to autodetect.
	If possible will attempt to read as RGB or RGBA
	Otherwise, will attempt to read Z channel. 
	If given, channels should be given as tuple of strings
	Returns: A numpy array of shape (height, width, number of channels)
	"""
	if type(file) == str:
		file = OpenEXR.InputFile(file)
	dw = file.header()['dataWindow']
	width = dw.max.x - dw.min.x + 1
	height = dw.max.y - dw.min.y + 1
	FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
	if channels is None:
		channels = autodetect_channels(file.header())
	image = np.array([array.array('f', file.channel(Chan, FLOAT)) for Chan in channels])
	image = np.reshape(image.T, (height, width, len(channels)))
	return image
def write_image(filename, image, channels = None):
	"""
	Write the exr image to the specified filename. If no channels are given, write_image will guess
	"""
	if channels is None:
		if image.shape[2] == 1:
			channels = ("z")
		elif image.shape[2] == 3:
			channels = ("R", "G", "B")
		elif image.shape[2] == 4:
			channels = ("R", "G", "B", "A")
		else:
			assert False, "Unable to guess channel names"
	splitted = [layer.flatten() for layer in np.split(image, image.shape[2], 2)]
	channel_strings = [ array.array('f', Chan).tostring() for Chan in splitted ]
	ch_dict = {}
	for ch_name, ch_string in zip(channels, channel_strings):
		ch_dict[ch_name] = ch_string
	# Write the three color channels to the output file
	header = OpenEXR.Header(image.shape[1], image.shape[0])
	if channels == ("R", "G", "B", "A"):
		header["channels"]["A"] = header["channels"]["R"]
	out = OpenEXR.OutputFile(filename, header)
	out.writePixels(ch_dict)
	
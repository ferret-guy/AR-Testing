import cv2
import numpy as np

from artag import ARtag


def display_scaled_image(name, image, scale):
	"""Function to display a scaled cv2 image
	:param name:
		Window name
	:type name:
		basestring
	:param image:
		Image as numpy array
	:type image:
		numpy.ndarray
	:param scale:
		Scale factor applied to image
	:type scale:
		float
	"""
	height, width = image.shape[:2]
	cv2.imshow(name, cv2.resize(image,
			(int(scale * width), int(scale * height)),
			interpolation=cv2.INTER_CUBIC))


def mask_black(img):
	"""Function to mask for black boxes in an BGR image
	:param img:
		Numpy image to process
	:type img:
		numpy.ndarray
	:return:
		the masked image
	:rtype:
		numpy.ndarray
	"""
	return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
						cv2.THRESH_BINARY, 511, 10)


def find_tag(img):
	"""
	Identify AR tags in an image

	:param img:
		The image to be processed, may be pre-converted to greyscale
	:type img:
		numpy.ndarray
	:return:
		Tuple[List[artag], numpy.ndarray]
	"""
	# Convert the image to greyscale if it is not
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dims = img.shape[:2]
	# Normalize image size
	img, _ = rescale_img(img)
	kernel = np.ones((3, 3), np.uint8)
	img = mask_black(img)
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Clean up contours to regular shapes
	culled_conts = []
	for c, h in zip(contours, hierarchy[0]):
		c = cv2.convexHull(c)
		epsilon = 0.01 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
		culled_conts.append((approx, h))

	img, scale = rescale_img(img, dims)

	# Identify nested shapes as possible AR tags
	ar_tags = []
	for c, h in culled_conts:
		# 4 sided tags
		if len(c) == 4:
			# The tag contains a smaller contour
			if h[2] != -1:
				tag = ARtag(c, culled_conts[h[2]][0], scale)
				if tag.valid():
					ar_tags.append(tag)

	img = cv2.merge((img, img, img))

	return ar_tags, img


def rescale_img(img, dims=None, width=1024):
	"""Rescale image, either to the dimensions provided or to the width
	provided, returning the image and it's dimensions

	:param img:
		The input image
	:type img:
		numpy.ndarray
	:param dims:
		Provide this parameter to fit the image to a specific size
	:type dims:
		Tuple[int, int]
	:param width:
		The width to resize the image to, (overridden by dims) deafualts to 1024
	:type width:
		int
	:return:
		Tuple of image and its dimensions
	:rtype:
		Tuple[numpy.ndarray, Tuple[int, int]]
	"""
	grey = img
	if dims is None:
		r = float(width) / grey.shape[1]
		dims = (width, int(grey.shape[0] * r))
	else:
		dims = dims[::-1]
	scale = (dims[1] / float(grey.shape[0]), dims[0] / float(grey.shape[1]))
	return cv2.resize(grey, dims, interpolation=cv2.INTER_AREA), scale


if __name__ == "__main__":
	# Display results for all test images
	import os
	for f in list(os.walk("Test imges"))[0][2]:
		if "jpg" in f:
			in_img = cv2.imread("Test imges/" + f)
			grey = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
			ar_tags, proc_img = find_tag(grey)
			for i in ar_tags:
				proc_img = i.draw(proc_img)
				in_img = i.draw(in_img)
			display_scaled_image("Image - {}".format(f), proc_img, .25)
			display_scaled_image("Orig Image - {}".format(f), in_img, .25)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

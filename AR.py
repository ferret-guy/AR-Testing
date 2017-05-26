import numpy as np
import cv2

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


def mask_blask(img):
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
	sensitivity = 70
	lower_white = np.array([0, 0, 0])
	upper_white = np.array([255, 255, sensitivity])
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_white, upper_white)
	return mask

def find_tag(img):
	"""Function to find tags in an image"""
	kernel = np.ones((5, 5), np.uint8)
	bigkernel = np.ones((10, 10), np.uint8)

	img = mask_blask(img)

	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img = cv2.dilate(img, bigkernel, iterations=2)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, bigkernel)

	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Clean up contours to regular shapes
	culled_conts = []
	for c, h in zip(contours, hierarchy[0]):
		c = cv2.convexHull(c)
		epsilon = 0.1 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
		culled_conts.append((approx, h))

	# Identify nested shapes as possible AR tags
	ar_tags = []
	for c, h in culled_conts:
		if len(c) == 4:
			if h[2] != -1:
				ar_tags.append(ARtag(c, culled_conts[h[2]][0]))

	ar_tags = sorted(ar_tags, key=lambda x: cv2.contourArea(x.outer_cont), reverse=True)

	img = cv2.merge((img, img, img))

	for i in ar_tags:
		img = i.draw(img)
		break

	return img


if __name__ == "__main__":
	in_img = cv2.imread("Test imges/10.jpg")
	proc_img = find_tag(in_img)
	display_scaled_image("Image", proc_img, .25)
	display_scaled_image("Orig Image", in_img, .25)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

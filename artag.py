import math

import cv2


class ARtag(object):
	def __init__(self, outer, inner, scale):
		"""Create a ARTag object, based on its inner contour,
		outer contour and the scale of the image processed as
		related to the image that was processed to extract them

		:param outer:
			The outer contour (Assumed to have 4 sides)
		:param inner:
			The outer contour (Assumed to have at least 4 sides)
		:param scale:
			The scale of the image that was processed to create this tag
		:type scale:
			Tuple[int, int]
		"""
		self.outer_cont = outer
		for i, c in enumerate(self.outer_cont):
			self.outer_cont[i][0][0] = c[0][0] * scale[1]
			self.outer_cont[i][0][1] = c[0][1] * scale[0]
		self.inner_cont = inner
		for i, c in enumerate(self.inner_cont):
			self.inner_cont[i][0][0] = c[0][0] * scale[1]
			self.inner_cont[i][0][1] = c[0][1] * scale[0]

		self.outer_tup = [tuple(i[0]) for i in outer]
		self.inner_tup = [tuple(i[0]) for i in inner]

		x, y = zip(*self.inner_tup)
		x = sum(x) / len(x)
		y = sum(y) / len(y)

		self.outer_tup = sorted(self.outer_tup, key=lambda p: math.sqrt((p[0] - x)**2 + (p[1] - y)**2))

		# Points of interest labeled clockwise from marked corner
		self.p0 = self.outer_tup[0]
		self.p1 = self.outer_tup[1]
		self.p2 = self.outer_tup[3]
		# Furthest point must be the 3rd point
		self.p3 = self.outer_tup[2]

	def draw(self, img):
		"""Draw the tag on a 3 channel image

		:param img:
			The 3 channel image
		:type img:
			numpy.ndarray
		:return:
			The input image
		:rtype:
			numpy.ndarray
		"""
		img = cv2.drawContours(img, [self.outer_cont], -1, (0, 255, 0), 10)
		img = cv2.drawContours(img, [self.inner_cont], -1, (255, 0, 0), 10)
		img = cv2.circle(img, tuple(self.p0), 20, (0, 0, 255), 10)
		return img

	def ratio(self, img):
		"""Ratio of the size of the Tag vs the input image size

		:param img:
			The input image
		:type img:
			numpy.ndarray
		:return:
			The processed image
		:rtype:
			numpy.ndarray
		"""
		return cv2.contourArea(self.outer_cont) / float(img.shape[0] * img.shape[1])

	def in_outer(self):
		return cv2.contourArea(self.inner_cont) / cv2.contourArea(self.outer_cont)

	@staticmethod
	def dist(p0, p1):
		""" Return the distance between two tuples in the format (x, y)

		:param p0:
			The first point
		:type p0:
			Tuple[int, int]
		:param p1:
			The second point
		:type p1:
			Tuple[int, int]
		:return:
			The distance
		:rtype:
			float
		"""
		return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

	def valid(self):
		"""Check the validity of this artag
		Check the following items:
		 - Any line distance is more than 1.8 times the average length
		 - Any line distance is less than .2 times the average length
		 - The inner contour takes up more than 10% of the tag
		 - The inner contour takes up less than 1% of the tag

		:return:
			The validity of the tag
		:rtype:
			bool
		"""
		lengths = [
			self.dist(self.p0, self.p1),
			self.dist(self.p1, self.p2),
			self.dist(self.p2, self.p3),
			self.dist(self.p3, self.p0)
		]
		# check aspect ratio
		if any([i > 1.8 * sum(lengths) / len(lengths) for i in lengths]):
			return False
		if any([i < .2 * sum(lengths) / len(lengths) for i in lengths]):
			return False
		# Check area ratios
		if cv2.contourArea(self.inner_cont) / cv2.contourArea(self.outer_cont) > 0.1:
			return False
		if cv2.contourArea(self.inner_cont) / cv2.contourArea(self.outer_cont) < 0.01:
			return False
		return True

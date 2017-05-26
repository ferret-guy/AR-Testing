import numpy as np
import math
import cv2

class ARtag(object):
	def __init__(self, outer, inner):
		self.outer_cont = outer
		self.inner_cont = inner

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
		img = cv2.drawContours(img, [self.outer_cont], -1, (0, 255, 0), 10)
		img = cv2.drawContours(img, [self.inner_cont], -1, (255, 0, 0), 10)
		img = cv2.circle(img, tuple(self.p0), 20, (0, 0, 255), 10)
		return img
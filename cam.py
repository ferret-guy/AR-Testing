import cv2

import AR

if __name__ == '__main__':
	cap = cv2.VideoCapture(2)
	cap.set(3, 1920)
	cap.set(4, 1080)
	while True:
		_, in_img = cap.read()
		grey = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
		ar_tags, proc_img = AR.find_tag(grey)
		for i in ar_tags:
			proc_img = i.draw(proc_img)
			in_img = i.draw(in_img)
		AR.display_scaled_image("Image", proc_img, .5)
		AR.display_scaled_image("Orig Image", in_img, .5)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

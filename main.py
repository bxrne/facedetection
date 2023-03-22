import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("video.mp4") # stock video from pexels

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 640, 480)
cv2.setWindowTitle('img', 'Face Detection')

max_faces = 0

while cv2.waitKey(30) & 0xff != 27:
	for i in range(2): # skip 2 frames
		_, img = cap.read()

	if img is None: # loop video
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		continue

	img = cv2.resize(img, fx=0.5, fy=0.5, dsize=None)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale -> faster processing
	gray = cv2.GaussianBlur(gray, (5, 5), 0) # blur image -> reduce noise
	gray = cv2.equalizeHist(gray) # equalize histogram -> improve contrast

	faces = detector.detectMultiScale(gray, 1.1, 4)

	for (x, y, w, h) in faces:
		cv2.circle(img, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
		cv2.circle(img, (x + w//2, y + h//2), w//2, (0, 0, 255), 2)

	if len(faces) > max_faces:
		max_faces = len(faces)

	cv2.putText(img, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	cv2.putText(img, f'Max Faces: {max_faces}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	if len(faces) == max_faces:
		cv2.imwrite('max_faces.jpg', img)

	cv2.imshow('img', img)

cap.release()
cv2.destroyAllWindows()

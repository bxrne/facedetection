import cv2

max_faces = 0

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.setWindowTitle('img', 'Face Detection')
cv2.resizeWindow('img', 800, 600)
cv2.moveWindow('img', 0, 0)


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("video.mp4") # stock video from pexels
fps = cap.get(cv2.CAP_PROP_FPS)


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


	if len(faces) > max_faces:
		max_faces = len(faces)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

	cv2.putText(img, f'Faces: {len(faces)}', (10, 525), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.putText(img, f'{fps:.2f}fps', (800, 525), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	if len(faces) == max_faces:
		cv2.imwrite('max_faces.jpg', img)

	cv2.imshow('img', img)

cap.release()
cv2.destroyAllWindows()

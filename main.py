import cv2

choice = input("Video or live? (0 or 1): ")

if choice == "0":
	cap = cv2.VideoCapture(0)

elif choice == "1":
	cap = cv2.VideoCapture(1)
else:
	print("Invalid input, exiting...")
	exit()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("video.mp4") # stock video from pexels

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 640, 480)
cv2.setWindowTitle('img', 'Face Detection')

while cv2.waitKey(30) & 0xff != 27: 
	for i in range(2):
		_, img = cap.read()

	if img is None: # loop video
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		continue

	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	cv2.putText(img, "Faces: " + str(len(faces)), (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

	cv2.imshow('img', img)

cap.release()



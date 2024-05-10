import cv2

# img = cv2.imread('images/Peoples_1.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = cv2.CascadeClassifier('Faces.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 300)

while True:
    success, img = cap.read()
    img_ref = cv2.flip(img, 1)
    gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=4)

    for (x, y, w, h) in results:
        cv2.rectangle(img_ref, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)

    cv2.imshow('Video', img_ref)


    if cv2.waitKey(28) & 0xFF == ord('q'):
        break






# results = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
#
#
# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=1)
#
#
#
#
# cv2.imshow('res', img)
# cv2.waitKey(0)

# cap = cv2.VideoCapture('videos/vehicles.mp4')
# cap = cv2.VideoCapture(0)
# cap.set(3, 500)
# cap.set(4, 300)
# while True:
#     success, img = cap.read()
#     cv2.imshow('Video', img)
#
#     if cv2.waitKey(28) & 0xFF == ord('q'):
#         break
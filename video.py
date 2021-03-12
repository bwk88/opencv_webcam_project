import cv2 , time

first_frame = None
status_list=[]

video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cv2.waitKey(2000)

while True:
    status=0
    check , frame = video.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(25,25),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame = cv2.threshold(delta_frame,15,255,cv2.THRESH_BINARY_INV)[1]
    thresh_dilate = cv2.dilate(thresh_frame, None , iterations = 2)

    contours, hierarchy=cv2.findContours( thresh_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    for contour in contours :
        if cv2.contourArea(contour) < 10000 :
            continue
        status = 1
        ( x,y,w,h )=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)

    cv2.imshow("capture",gray)
    cv2.imshow("diff",delta_frame)
    cv2.imshow("threshold",thresh_dilate)
    cv2.imshow("colourFrame",frame)

    key=cv2.waitKey(1)
    print(gray)
    print(delta_frame)
    print(thresh_frame)

    if key==ord('q'):
        break

print(status_list)

video.release()
cv2.destroyAllWindows()

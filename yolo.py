
# yolo object detection Opencv with Python
# opencv framework

import cv2
import numpy as np
import time

fallStatus = False
frame_number = 0

# centerX, centerY为中点坐标, (right, bottom)为右下角(x, y)
def isFall(centerX ,centerY ,right ,bottom):
    global frame_number
    start = time.clock()  # 获取时间间隔
    Area_start = float((right - centerX) * (centerY - bottom))  # 获取变化的面积
    thresh = 1.37 * (right - centerX)  # VT一般在1.21m/s-2.05m/s   1.37为竖直方向变化速度
    print('framenumber:',str(frame_number))
    if frame_number % 10 == 1:
        now = time.clock()
        Area_now = float((right - centerX) * (centerY - bottom))
        SpineV = float(1000 * (Area_now - Area_start) / (now - start))  # 求得当前速度
        print('SpineV: '+str(SpineV)+'|'+'thresh: '+str(thresh))
        if(SpineV > thresh):  # 当前速度和阈值作比较
            print('人体重心下移的速度为 ' ,SpineV ,'m/s')
            return True
        else:
            return False

# load yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers_names = net.getLayerNames()
output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes),3))

#loading image
cap = cv2.VideoCapture(0)

while True:
    frame_number += 1

    _, frame = cap.read()

    height, width, channels = frame.shape

    # Detecting object
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop = False)

    #for b in blob:
    #    for n, img_blob in enumerate(b):
    #        cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #object detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x, center_y), 10, (0, 255, 0), 2)
                # rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    number_objects_deteted = len(boxes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x,y),(x+w, y+h),color, 2)
            cv2.putText(frame, label, (x, y+30), font, 2,color,3)

            # 当为人的时候调用isFall
            if label == 'person':
                fallStatus = isFall(center_x, center_y, x + w, x + h)



    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
    cv2.imshow("Image", frame)


    if fallStatus:
        print("fall")

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
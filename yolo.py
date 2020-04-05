
# yolo object detection Opencv with Python
# opencv framework

import cv2
import numpy as np
import time
import threading





# isFall线程函数，不断使用personBox变量来判断
#判断单人版本
def isFall():
    global  personBox
    global frame_number
    time.sleep(1)   #睡一秒
    print("跌倒检测线程已启动")
    x,y,w,h = personBox
    right = x+w
    bottom = y+h
    centerY = y+h/2
    centerX = x+w/2
    start = time.clock()  # 获取时间间隔
    Area_start = float((right - centerX) * (bottom - centerY))  # 获取变化的面积
    thresh_start = 1.37 * (right - centerX)  # VT一般在1.21m/s-2.05m/s   1.37为竖直方向变化速度

    while True:
        #print('framenumber:', str(frame_number))
        time.sleep(0.1)
        if time.clock() - start >= 0.5:#每差不多半秒出一点秒进行判断
            #记录时间
            now = time.clock()
            frame_number = 0
            #获取最新的人物框架的xywh，即为当前
            x, y, w, h = personBox
            right = x + w
            bottom = y + h
            centerY = y + h / 2
            centerX = x + w / 2

            thresh = thresh_start
            Area_now = float((right - centerX) * (bottom - centerY))
            print("now",str(now),"|start",str(start))
            print("Area_now",str(Area_now),"|Area_start",str(Area_start))
            SpineV = float(1000 * (Area_now - Area_start) / (now - start))  # 求得当前速度
            print('SpineV: '+str(SpineV)+'|'+'thresh: '+str(thresh))

            if(SpineV > thresh):  # 当前速度和阈值作比较
                print('人体重心下移的速度为 ' ,SpineV ,'m/s')

            #再次获取最新的人物框架的xywh，即为下次初值
            x, y, w, h = personBox
            right = x + w
            bottom = y + h
            centerY = y + h / 2
            centerX = x + w / 2
            start = time.clock()  # 获取时间间隔
            Area_start = float((right - centerX) * (bottom - centerY))  # 获取变化的面积
            thresh_start = 1.37 * (right - centerX)  # VT一般在1.21m/s-2.05m/s   1.37为竖直方向变化速

fallStatus = False
personBox = [0,0,0,0]#单人
frame_number = 0


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
print("摄像头启动")

print("准备启动跌倒检测线程")
t1 = threading.Thread(target=isFall)
t1.start()


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
            if label == 'person':#判断多人时，此处应该为append
                personBox = boxes[i]



    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
    cv2.imshow("Image", frame)


    if fallStatus:
        print("fall")

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
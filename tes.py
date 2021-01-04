import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.image_data_format()
from keras.models import model_from_json

#LOAD CNN MODEL
json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_final.h5")

'''
#GET IMAGE FROM CAMERA
cam = cv2.VideoCapture(0)
cv2.namedWindow("take photo")
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("take photo", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        imn = img_name
        img_counter += 1
        break
cam.release()
cv2.destroyAllWindows()

# cv2.imshow('photo taken', frame)
cv2.waitKey(0)

print(imn)
'''

#PICTURE PATH
train_data=[]
pict = 'correct.png'
# pict = 'wrong.png'
img = cv2.imread(pict,cv2.IMREAD_GRAYSCALE)

if img is not None:
	img=~img
	ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	img_dilate = cv2.dilate(thresh, np.ones((3,3), np.uint8), iterations=1)
	img_erode1 = cv2.erode(img_dilate, np.ones((4, 4), np.uint8), iterations=1)

	# FIND CONTOURS
	ret, ctrs, ret=cv2.findContours(img_erode1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	w=int(28)
	h=int(28)
    #print(len(cnt))
	rects=[]
	# FIND DIGITS
	for c in cnt :
		x,y,w,h= cv2.boundingRect(c)
		rect=[x,y,w,h]
		rects.append(rect)
	bool_rect=[]
	for r in rects:
		l=[]
		for rec in rects:
			flag=0
			if rec!=r:
				if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
					flag=1
				l.append(flag)
			if rec==r:
				l.append(0)
		bool_rect.append(l)
    #print(bool_rect)
	dump_rect=[]
	for i in range(0,len(cnt)):
		for j in range(0,len(cnt)):
			if bool_rect[i][j]==1:
				area1=rects[i][2]*rects[i][3]
				area2=rects[j][2]*rects[j][3]
				if(area1==min(area1,area2)):
					dump_rect.append(rects[i])
    #print(len(dump_rect)) 
	final_rect=[i for i in rects if i not in dump_rect]
    #print(final_rect)
    # GET LIST OF DIGITS ON PICTURE
	for r in final_rect:
		x=r[0]
		y=r[1]
		w=r[2]
		h=r[3]
		im_crop =thresh[y:y+h+10,x:x+w+10]
        
		im_resize = cv2.resize(im_crop,(28,28))
		im_resize=np.reshape(im_resize,(1,28,28))
		train_data.append(im_resize)
	
#PREDICTING PART BY TRAINED CNN MODEL
s=''
for i in range(len(train_data)):
    train_data[i]=np.array(train_data[i])
    train_data[i]=train_data[i].reshape(1,1,28,28)
    result=loaded_model.predict_classes(train_data[i])
    if(result[0]==10):
        s=s+'-'
    if(result[0]==11):
        s=s+'+'
    if(result[0]==12):
        s=s+'*'
    if(result[0]==0):
        s=s+'0'
    if(result[0]==1):
        s=s+'1'
    if(result[0]==2):
        s=s+'2'
    if(result[0]==3):
        s=s+'3'
    if(result[0]==4):
        s=s+'4'
    if(result[0]==5):
        s=s+'5'
    if(result[0]==6):
        s=s+'6'
    if(result[0]==7):
        s=s+'7'
    if(result[0]==8):
        s=s+'8'
    if(result[0]==9):
        s=s+'9'
    
print('string: '+s)
g = ''
r = ''
for i in range(len(s)):
	if s[i] == '-':
		if s[i+1] == '-':
			# -- means =
			r = r + str(s[i+2:])
			break
	else:
		g = g + str(s[i]) #PART TO BE EVALUATED
# print(rects)

#GETTING THE BIG RECT COOR
lr = len(rects) #COUNT OF DIGITS AND SYMBOLS
xy1 = rects[0] #FIRST DIGIT
xy2 = rects[lr-1] #LAST DIGIT

x1 = xy1[0]
y1 = xy1[1]

x2 = xy2[0]+xy2[2]
y2 = xy2[1]+xy2[3]

#GREEN RECT IF CORRECT, RED RECT IF WRONG
img_res = cv2.imread(pict)

if str(eval(g)) == r:
	# print('correct')
	cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.imshow('result is correct', img_res)
	cv2.waitKey(0)
else:
	# print('wrong')
	cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 0, 255), 2)
	cv2.imshow('result is wrong', img_res)
	cv2.waitKey(0)
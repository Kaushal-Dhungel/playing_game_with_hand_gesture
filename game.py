from keras.models import load_model
import cv2
import numpy as np
from direct_key import PressKey, Right,Left,Up,Down
import time

REV_CLASS_MAP = {
    #"Up": 0,  # the game for which i made this model doesnt need Up button
    0:"Down",
    1:"Left",
    2:"none",
    3:"Right"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def play(gesture):
    if gesture == "Down":
        PressKey(Down)
        time.sleep(1)

    elif gesture == "Left":
        PressKey(Left)
        time.sleep(1)
     
    elif gesture == "Right":
        PressKey(Right)
        time.sleep(1)

model = load_model("car-racing-model.h5")

cap = cv2.VideoCapture(0)
#you can specify the window height and width from here, I choosed the default
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)

while(True):
    ret,frame = cap.read()
    
    if not ret:
        continue
    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    #region of interest
    roi = frame[100:400,100:400]
    img = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    
    # prediction
    pred = model.predict(np.array([img]))
    gesture_code = np.argmax(pred[0])
    user_gesture = mapper(gesture_code)

    #start_playing
    play(user_gesture)

    #putting the user_geture on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,user_gesture,(50,50),font,1.2,(255,0,0),2)

    cv2.imshow('window',frame)

    if cv2.waitKey(27) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
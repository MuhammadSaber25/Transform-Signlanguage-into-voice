import requests
import numpy as np
import cv2
############################################################
import os
from imageai.Prediction.Custom import CustomImagePrediction
#Loading model
predictor = CustomImagePrediction()
predictor.setModelPath(model_path="model.h5")
predictor.setJsonPath(model_json="model.json")
predictor.loadFullModel(num_objects=11)
###########################################################
def check():
    if prediction[0]=="aywa":
     playsound('SignLanguage\Voiceaywa.wav')
    elif prediction[0]=="no":
     playsound('SignLanguage\Voiceno.wav')
    elif prediction[0]=="dont":
     playsound('SignLanguage\Voicedont.wav')
    elif prediction[0]=="shokrn":
     playsound('SignLanguage\Voiceshokrn.wav')
    elif prediction[0]=="sorry":
     playsound('SignLanguage\Voicesorry.wav')
    elif prediction[0]=="tmam":
     playsound('SignLanguage\Voicetmam.wav')
 ############################################################
img_counter = 0
while True:
    img_res = requests.get("http://192.168.1.2:8080/shot.jpg")
    img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr,-1)

    sky = img[0:420, 0:420]
    cv2.imshow(' sky !', sky)
    
    k = cv2.waitKey(1)
    if k%256 == 32:
        img_name = "image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, sky)
        print("{} written!".format(img_name))
        prediction, probability = predictor.predictImage(image_input=os.path.join(os.getcwd(), img_name), result_count=1)
        print(prediction, " :", probability)
        #check()
        os.remove(img_name)
        img_counter += 1
        
    elif k%256 == 9:
        img2=cv2.flip(sky,1)
        cv2.imshow(' flip !!', img2)
        
    elif k%256 == 97:    
        img_name = "image2_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, img2)
        print("{} written2!".format(img_name))
        prediction, probability = predictor.predictImage(image_input=os.path.join(os.getcwd(), img_name), result_count=1)
        print(prediction, " :", probability)
        #check()
        os.remove(img_name)
        img_counter += 1

    elif k%256 == 27:
        break
    
img_res.destroy()
cv2.destroyAllWindows()
            
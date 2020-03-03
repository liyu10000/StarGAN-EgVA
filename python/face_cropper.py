import cv2
import sys
import os


class FaceCropper(object):
    # CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_default.xml"
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        
    def generate(self, image_path, save_path, show_result):
        os.makedirs(save_path, exist_ok=True)
        file_list = []
        if os.path.isfile(image_path):
            file_list = [image_path]
        else:
            file_list = os.listdir(image_path)
        
        for file in file_list:
            file_path = os.path.join(image_path, file)
            img = cv2.imread(file_path)
            if (img is None):
                print("Can't open image file:", file)
                continue

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(img, 1.3, 5, minSize=(100, 100))
            if (faces is None):
                print('Failed to detect face')
                continue

            if (show_result):
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            facecnt = len(faces)
            print("Detected faces: {} for {}".format(facecnt, file))
            
            save_name = os.path.join(save_path, os.path.splitext(file)[0]+'.png')
            for (x, y, w, h) in faces:
                faceimg = img[y:y+h, x:x+w]
                saveimg = cv2.resize(faceimg, (128, 128))
                cv2.imwrite(save_name, saveimg)


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    if (argc != 3):
        print('Usage: %s [image file]' % args[0])
        quit()

    detecter = FaceCropper()
    detecter.generate(args[1], args[2], False)
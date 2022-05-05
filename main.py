from tkinter import *
from Features import *
from threading import Thread

from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics

import dlib, cv2, os, csv
import pandas as pd
import numpy as np
import threading
csv_lock = threading.Lock()
from multiprocessing.pool import ThreadPool as Pool

global clf
global counter, label
counter = 0
totalCounter = 0

def selectFolder():
    return filedialog.askdirectory()

def selectFile():
    return filedialog.askopenfilename()

def annotate_img(sample, landmarks):
    for n in range(0, 68):
        cv2.circle(sample, (landmarks.part(n).x, landmarks.part(n).y), 1, (0, 0, 255), -1)
    return sample

def engineer_features(landmarks, label):
    f = Features(landmarks)
    f_vector = f.run()
    f_vector.insert(0, label)
    return f_vector
    
def get_landmarks(sample, label):
    datFile = "shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    lm_extractor = dlib.shape_predictor(datFile)

    faces = face_detector(sample)
    for face in faces:
        f_vector = []
        landmarks = lm_extractor(sample, face)
        f_vector = engineer_features(landmarks, label)

        if (label != "test"):
            if not os.path.exists('landmarks.csv'):
                with open('landmarks.csv', 'w', newline='') as csvfile:
                    with csv_lock:
                        writer = csv.writer(csvfile)
                        writer.writerow(['label', 'left_eyebrow_length', 'right_eyebrow_length', 'lip_width', 'lip_height', 'left_eye_height', 'right_eye_height', 'left_eye_width', 'right_eye_width'])
                        writer.writerow(f_vector)
            else:
                with open('landmarks.csv', 'a', newline='') as csvfile:
                    with csv_lock:
                        writer = csv.writer(csvfile)
                        writer.writerow(f_vector)
        else:
            return f_vector
        
        
        
        #annotated_sample = annotate_img(sample, landmarks)
        # cv2.imshow("Annotated", annotated_sample)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def thread_extract():
    thread = Thread(target=extract)
    thread.start()

def test_single():
    global clf
    t = selectFile()
    sample = cv2.imread(t)
    sample = cv2.resize(sample, (640, 490), interpolation= cv2.INTER_LINEAR)
    

def features(sample, f):
    global counter
    try:
        get_landmarks(sample, f)
    except:
        print('error extracting features')
    counter+=1




def extract():
    print("Extracting features from dataset...")
    dataset_path = "C:/Users/ewans/Desktop/cmp304-cwk2/set"

    sample = []
    f = []

    global counter, label
    counter = 0
    totalCounter = 0

    for folder in os.listdir(dataset_path):
        for image in os.listdir(dataset_path + '/' + folder):
             sample.append(cv2.imread(dataset_path + '/' + folder + '/' + image))
             f.append(folder)
             totalCounter += 1

    pool = Pool()
    for i in range(len(sample)):
        pool.apply_async(features, args=(sample[i], f[i]))
        label.config(text=str(counter) + "/" + str(totalCounter))
        label.pack()
        window.update()

    window.update()
    label.pack_forget()

    pool.close()
    pool.join()




def train():
    global clf
    print("Training the model...")
    
    df = pd.read_csv('landmarks.csv')
    df = df.drop(df.index[0])
    #encode the labels
    le = preprocessing.LabelEncoder()
    le.fit(df['label'])
    df['label'] = le.transform(df['label'])
    
    x_train, x_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, shuffle=True, random_state=42)

    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    clf = SVC(kernel='linear', C=1.0, random_state=0)
    clf.fit(x_train, y_train)




    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))

    #print(metrics.classification_report(y_test, y_pred))

    #print(metrics.confusion_matrix(y_test, y_pred))



def createWindow():
    window = Tk()
    window.title("CMP304 Coursework 2 - Emotion Recognition")
    window.geometry("500x500")
    window.configure(background='gray')

    label = Label(window, text="Emotion Recognition", fg="white", bg="gray", font=("Helvetica", 16))
    label.pack()

    extractButton = Button(window, text="Extract dataset features to CSV", command=thread_extract)
    extractButton.pack()
    extractButton.place(relx=0.5, rely=0.5, anchor=CENTER)

    trainButton = Button(window, text="Train from CSV", command=train)
    trainButton.pack()
    trainButton.place(relx=0.5, rely=0.6, anchor=CENTER)

    testButtonSingle = Button(window, text="Test the model on a single image", command=test_single)
    testButtonSingle.pack()
    testButtonSingle.place(relx=0.5, rely=0.7, anchor=CENTER)

    #text at bottom of window
    label2 = Label(window, text="Ewan Stewart - 1900598", fg="white", bg="gray", font=("Helvetica", 12))
    label2.pack()
    label2.place(relx=0.5, rely=0.9, anchor=CENTER)

    return window

def main():
    global window, counter, totalCounter, label
    window = createWindow()
    label = Label(window, text=str(counter) + "/" + str(totalCounter), fg="white", bg= "gray", font=("Helvetica", 16))  
    window.mainloop()



if __name__ == "__main__":
    main()

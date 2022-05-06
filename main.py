#imports
from tkinter import *
from Features import *
import dlib, cv2, os, csv
import pandas as pd
import numpy as np
import threading

from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, metrics, preprocessing
from multiprocessing.pool import ThreadPool as Pool
from threading import Thread




#control access to the csv when multi threading the extraction
csv_lock = threading.Lock()

global clf, counter, label
counter, totalCounter = 0

def selectFolder(): #get the user to select the folder containing the dataset
    return filedialog.askdirectory()

def selectFile(): #get the user to select an individual image to test the model
    return filedialog.askopenfilename()

def annotate_img(sample, landmarks):    #visualise the landmarks on the image
    for n in range(0, 68):
        cv2.circle(sample, (landmarks.part(n).x, landmarks.part(n).y), 1, (0, 0, 255), -1)
    return sample

def engineer_features(landmarks, label):    #extract the landmarks for each image and return a vector of features with the emotion
    f = Features(landmarks)                 #create a feature object which handles which feature engineering 
    f_vector = f.run()                      #start extraction in feature obj
    f_vector.insert(0, label)               #insert label at the start of hte feature vector
    return f_vector
    
def get_landmarks(sample, label):   #using a provided image extract the landmarks from the image using the shape predictor
    datFile = "shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    lm_extractor = dlib.shape_predictor(datFile)

    faces = face_detector(sample)
    for face in faces:
        f_vector = []
        landmarks = lm_extractor(sample, face)
        f_vector = engineer_features(landmarks, label)  #get the features for the image

        if (label != "test"):                                           #if the image is being used for training: write to the csv file
            if not os.path.exists('landmarks.csv'):                     #if the file doesn't exist create it
                with open('landmarks.csv', 'w', newline='') as csvfile: #open the landmark csv file
                    with csv_lock:                                      #thread safe lock
                        writer = csv.writer(csvfile)                    #write the column headers if first time, write the feature vector
                        writer.writerow(['label', 'left_eyebrow_length', 'right_eyebrow_length', 'lip_width', 'lip_height', 'left_eye_height', 'right_eye_height', 'left_eye_width', 'right_eye_width'])
                        writer.writerow(f_vector)
            else:                                                       #if file exists append to end
                with open('landmarks.csv', 'a', newline='') as csvfile:
                    with csv_lock:
                        writer = csv.writer(csvfile)
                        writer.writerow(f_vector)
        else:
            return f_vector         #if the image is being used for testing: return the feature vector only
        

        #uncomment to see the annotated images
        
        #annotated_sample = annotate_img(sample, landmarks)
        # cv2.imshow("Annotated", annotated_sample)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def thread_extract():   #start the extract function in a thread so the GUI doesn't freeze
    thread = Thread(target=extract)
    thread.start()

def test_single():
    global clf              #use the trained model to predict the emotion of the image
    t = selectFile()        #get the user to select an image
    sample = cv2.imread(t)
    sample = cv2.resize(sample, (640, 490), interpolation= cv2.INTER_LINEAR)    #pre-processing re-size the image
    

def features(sample, f):
    global counter  #counter for displaying processing progress on the GUI

    try:
        get_landmarks(sample, f)    #get the landmarks for the image
    except:
        print('error extracting features')

    counter+=1




def extract():  #extraction function which provides file paths and labels for the model to train
    global counter, label

    print("Extracting features from dataset...")

    dataset_path = "C:/Users/ewans/Desktop/cmp304-cwk2/set"     #path to the datset
    sample, labels = []                                         #array to store the image paths and labels
    counter, totalCounter = 0                                   #image processing progress counter

    for folder in os.listdir(dataset_path):                     #for each folder in the dataset
        for image in os.listdir(dataset_path + '/' + folder):   #for each image in the labelled folder
             sample.append(cv2.imread(dataset_path + '/' + folder + '/' + image))   #append the image to the sample array
             labels.append(folder)                                                  #append the emotion of that image
             totalCounter += 1                                  #counter to determine how many images are in the dataset

    pool = Pool()       #start a pool to thread the extraction progress
    for i in range(len(sample)):
        pool.apply_async(features, args=(sample[i], labels[i]))
        label.config(text=str(counter) + "/" + str(totalCounter))   #update progress of the extraction
        label.pack()                                                #display label 
        window.update()                                             #update the window to show label

    window.update()
    label.pack_forget() #delete label after extraction is complete

    pool.close()
    pool.join()




def train():
    global clf
    print("Training the model...")
    
    df = pd.read_csv('landmarks.csv')       #read the feature csv file 
    df = df.drop(df.index[0])               #remove the column headers
    le = preprocessing.LabelEncoder()       #create a label encoder
    le.fit(df['label'])                     #encode the labels
    df['label'] = le.transform(df['label']) #replace the labels with the encoded labels
    
    x_train, x_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'],
    test_size=0.2, shuffle=True, random_state=42)   

    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    clf = SVC(kernel='linear', C=1.0, random_state=0)   #create a linear SVM classifier
    clf.fit(x_train, y_train)                           #train the model using the provided features and labels

    print(clf.score(x_train, y_train))  #print the accuracy of the training
    print(clf.score(x_test, y_test))    #print the accuracy of the testing

    #print(metrics.classification_report(y_test, y_pred))
    #print(metrics.confusion_matrix(y_test, y_pred))



def createWindow(): #GUI window
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

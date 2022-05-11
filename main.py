#imports
from tkinter import *
from Features import *

import dlib, cv2, os, csv
import pandas as pd
import threading
import matplotlib.pyplot as plt

from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from multiprocessing.pool import ThreadPool as Pool
from threading import Thread

#control access to the csv when multi threading the extraction
csv_lock = threading.Lock()
csv_path = "engineeredFeatures.csv"
global counter, label
counter = 0
totalCounter = 0


def engineer_features(landmarks, label):    #extract the landmarks for each image and return a vector of features with the emotion
    f = Features(landmarks)                 #create a feature object which handles which feature engineering 
    f_vector = f.run()                      #start extraction in feature obj
    f_vector.insert(0, label)               #insert label at the start of hte feature vector
    
    return f_vector
    
def get_landmarks(sample, label):   #using a provided image extract the landmarks from the image using the shape predictor
    datFile = "shape_predictor_68_face_landmarks.dat"
    column_header = ['label', 'l_eb_l', 'r_eb_l', 'u_ll_l', 'u_rl_l', 'l_l', 'l_h']
    face_detector = dlib.get_frontal_face_detector()
    lm_extractor = dlib.shape_predictor(datFile)

    faces = face_detector(sample)
    for face in faces:
        f_vector = []
        landmarks = lm_extractor(sample, face)
        f_vector = engineer_features(landmarks, label)  #get the features for the image

        if (label != "test"):                                           #if the image is being used for training: write to the csv file
            if not os.path.exists(csv_path):                     #if the file doesn't exist create it
                with open(csv_path, 'w', newline='') as csvfile: #open the landmark csv file
                    with csv_lock:                                      #thread safe lock
                        writer = csv.writer(csvfile)                    #write the column headers if first time, write the feature vector
                        writer.writerow(column_header)
                        writer.writerow(f_vector)
            else:                                                       #if file exists append to end
                with open(csv_path, 'a', newline='') as csvfile:
                    with csv_lock:
                        writer = csv.writer(csvfile)
                        writer.writerow(f_vector)
        else:
            return f_vector         #if the image is being used for testing: return the feature vector only
        

def thread_extract():   #start the extract function in a thread so the GUI doesn't freeze
    thread = Thread(target=extract)
    thread.start()

def features(sample, f):
    global counter  #counter for displaying processing progress on the GUI

    try:
        get_landmarks(sample, f)    #get the landmarks for the image
    except Exception as e:
        print(e)
        
    counter+=1

def extract():  #extraction function which provides file paths and labels for the model to train
    global counter, label

    print("Extracting features from dataset...")

    dataset_path = "set"     #path to the datset
    sample = []
    labels = []                                         #array to store the image paths and labels
    counter = 0
    totalCounter = 0                                   #image processing progress counter

    for folder in os.listdir(dataset_path):                     #for each folder in the dataset
        for image in os.listdir(dataset_path + '/' + folder):   #for each image in the labelled folder
             sample.append(cv2.imread(dataset_path + '/' + folder + '/' + image))   #append the image to the sample array
             sample[counter] = cv2.resize(sample[counter], (640, 490), interpolation= cv2.INTER_LINEAR)
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
    print("Training the model...")
    
    df = pd.read_csv(csv_path)       #read the feature csv file 
    le = preprocessing.LabelEncoder()       #create a label encoder
    le.fit(df['label'])                     #encode the labels
    df['label'] = le.transform(df['label']) #replace the labels with the encoded labels
    
    x_train, x_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'],
    test_size=0.2, shuffle=True)   

    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    
    clf = SVC(kernel='linear', C=1, random_state=0)   #create a linear SVM classifier
    clf.fit(x_train, y_train)                           #train the model using the provided features and labels

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(x_train, y_train)

    import seaborn as sns

    print("Accuracy Training (SVM): " , clf.score(x_train, y_train))
    print("Accuracy Testing (SVM):", clf.score(x_test, y_test))

    print("Accuracy Training (KNN): " , knn.score(x_train, y_train))
    print("Accuracy Testing (KNN):", knn.score(x_test, y_test))

    print("Accuracy Training (RF): " , rf.score(x_train, y_train))
    print("Accuracy Testing (RF):", rf.score(x_test, y_test))

    svm_report = metrics.classification_report(y_test, clf.predict(x_test), target_names=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'], output_dict=True)
    knn_report = metrics.classification_report(y_test, knn.predict(x_test), target_names=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'], output_dict=True)
    rf_report = metrics.classification_report(y_test, rf.predict(x_test), target_names=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'], output_dict=True)
    
    

    sns.heatmap(pd.DataFrame(svm_report).iloc[:-1, :].T, annot=True)
    plt.title('SVM')
    plt.tight_layout()
    plt.show()

    sns.heatmap(pd.DataFrame(knn_report).iloc[:-1, :].T, annot=True)
    plt.title('KNN')
    plt.tight_layout()
    plt.show()

    sns.heatmap(pd.DataFrame(rf_report).iloc[:-1, :].T, annot=True)
    plt.title('RF')
    plt.tight_layout()
    plt.show()




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

    label2 = Label(window, text="Ewan Stewart - 1900598", fg="white", bg="gray", font=("Helvetica", 12))
    label2.pack()
    label2.place(relx=0.5, rely=0.9, anchor=CENTER)

    return window

def main():
    global window, counter, totalCounter, label
    window = createWindow()
    label = Label(window, text=str(counter) + "/" + str(totalCounter), fg="red", bg= "gray", font=("Helvetica", 16))  
    window.mainloop()

if __name__ == "__main__":
    main()

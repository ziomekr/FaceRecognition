import tkinter as tk
from tkinter import messagebox
from FaceRecognizer import FaceRecognizer

class MyDialog:
    #Pop up dialog for entering new username
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        self.myLabel = tk.Label(top, text='Enter new username')
        self.myLabel.pack()
        self.myEntryBox = tk.Entry(top)
        self.myEntryBox.pack()
        self.mySubmitButton = tk.Button(top, text='Submit', command=self.send)
        self.mySubmitButton.pack()

    def send(self):
        self.username = self.myEntryBox.get()
        self.top.destroy()

faceRecognizer = FaceRecognizer()

def onAddUserClick():
    #get new username and capture training data
    inputDialog = MyDialog(root)
    root.wait_window(inputDialog.top)
    if(len(inputDialog.username) == 0):
        messagebox.showinfo("Error!", "Username cannot be empty!")
    else:
        faceRecognizer.capture_dataset_from_cam(str(inputDialog.username))
        root.update()

def onTrainModelClick():
    #train model with created users
    faceRecognizer.train_recognizer()

def onStartButtonClick():
    #provide live webcam view with recognized faces
    faceRecognizer.live_prediction()

root = tk.Tk()
root.title("Face recognition app - AI Fundamentals course")

#Simple GUI initialization
addUserButton = tk.Button(root, text='Add new user', command=onAddUserClick, width=80)
addUserButton.pack()
trainModelButton = tk.Button(root, text='Train model', command=onTrainModelClick, width=80)
trainModelButton.pack()

startFaceRecognitionButton = tk.Button(root, text='Start face recognition', command=onStartButtonClick, width=80)
startFaceRecognitionButton.pack()

root.mainloop()
root.update()
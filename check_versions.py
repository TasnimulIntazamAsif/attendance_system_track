import platform, struct
import numpy as np
import cv2
import face_recognition
import dlib

print("Python arch:", struct.calcsize("P")*8, "bit")
print("Platform:", platform.platform())
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("dlib:", dlib.__version__)
print("face_recognition:", face_recognition.__version__)
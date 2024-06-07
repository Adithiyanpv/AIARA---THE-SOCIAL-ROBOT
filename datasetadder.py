import cv2
import os

cam = cv2.VideoCapture('./recorded-video.webm')
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# For each person, enter one numeric face id
# face_id = input('\n Enter user name and press <return>: ')
face_id = "pradeep"
# Create directory for dataset if it doesn't exist
dataset_dir = face_id
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Initialize individual sampling face count
count = 0

while count < 500:
    ret, img = cam.read()

    count += 1

    # Save the captured image into the datasets folder
    cv2.imwrite(f"./pradeep2/User_{face_id}_{count}.jpg", img)

    cv2.imshow('image', img)

    k = cv2.waitKey(5)  # wait for 100 milliseconds
    if k == 27:  # Press 'ESC' to exit
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

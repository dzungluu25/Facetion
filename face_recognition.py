import face_recognition
import cv2
import numpy as np
import os
from sklearn import neighbors
from sklearn.model_selection import train_test_split
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            # Assuming your image names contain the label, e.g., "person1.jpg"
            label = filename.split('_')[0]
            labels.append(label)
    return images, labels

# Load your face data (you can use your own dataset)
faces, labels = load_images_from_folder("face_data")  # Folder with face images
def encode_faces(faces): 
    face_encodings = []
    for face in faces:
        # Convert image from BGR to RGB (OpenCV loads as BGR)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Encode the face (128-dimension face encoding)
        encodings = face_recognition.face_encodings(face_rgb)
        if encodings:
            face_encodings.append(encodings[0])
    return face_encodings

# Encode faces
face_encodings = encode_faces(faces)
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(face_encodings, labels, test_size=0.2, random_state=42)

# Use K-Nearest Neighbors classifier for face recognition
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
knn_clf.fit(X_train, y_train)

# Test the model on test data
accuracy = knn_clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Capture video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    # Convert frame to RGBx
    rgb_frame = frame[:, :, ::-1]
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locaions(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Use the trained KNN classifier to predict the face
        matches = knn_clf.kneighbors([face_encoding], n_neighbors=1)
        name = matches[1][0][0] if matches else "Unknown"
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Label the face with the name
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()

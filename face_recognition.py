import face_recognition
import cv2
import numpy as np
from flask import Flask ,render_template,Response
# face recognition

app=Flask(__name__)

camera=cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
sayan_image = face_recognition.load_image_file("sayan/sayan.jpg")
sayan_face_encoding = face_recognition.face_encodings(sayan_image)[0]

# Load a second sample picture and learn how to recognize it.
shameek_image = face_recognition.load_image_file("shameek/shameek.jpg")
shameek_face_encoding = face_recognition.face_encodings(shameek_image)[0]

# Load a second sample picture and learn how to recognize it.
karfa_image = face_recognition.load_image_file("karfa/karfa.jpg")
karfa_face_encoding = face_recognition.face_encodings(karfa_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    sayan_face_encoding,
    shameek_face_encoding,
    karfa_face_encoding
]
known_face_names = [
    "sayan",
    "shameek",
    "karfa"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def genreate_video():
    while True:
        #reading the video data
        success,frame=camera.read()
        if not success:
            break
        else:

             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
             rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
             face_locations = face_recognition.face_locations(rgb_small_frame)
             face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
             face_names = []
             for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
             for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

             ret,buffer=cv2.imencode('.jpg',frame)
             frame=buffer.tobytes()
             yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    


@app.route('/')
def index():
    return render_template('live_cam.html')

@app.route('/video')
def video():
    return Response(genreate_video(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)    

import cv2
import face_recognition
from simple_facerec import Advanced_Facerec

class SimpleFacerecWrapper:
    def __init__(self, simple_facerec):
        self.simple_facerec = simple_facerec

    def is_known_face(self, name):
        return name in self.simple_facerec.known_face_names

    def detect_known_faces(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        known_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.simple_facerec.known_face_encodings, face_encoding)
            name = "Unknown"
            if any(matches):
                first_match_index = matches.index(True)
                name = self.simple_facerec.known_face_names[first_match_index]

            known_names.append(name)

        return face_locations, known_names

def main():
    # Use the wrapper class
    sfr = SimpleFacerecWrapper(Advanced_Facerec())
    sfr.simple_facerec.load_encoding_images("images/Furqan_alii/")
    sfr.simple_facerec.load_encoding_images("images/Farooq/")
    #sfr.simple_facerec.load_encoding_images("images/Fahad/")

    # Load Camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, camera_real = cap.read()

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(camera_real)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Check if the detected face is a known face
            is_known_face = sfr.is_known_face(name)

            color = (0, 0, 200)  # Default color for rectangles and text

            if is_known_face:
                cv2.putText(camera_real, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            else:
                cv2.putText(camera_real, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

            cv2.rectangle(camera_real, (x1, y1), (x2, y2), color, 4)

        cv2.imshow("FACE_CAM", camera_real)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

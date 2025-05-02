import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle
import imgaug.augmenters as iaa

class Advanced_Facerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing
        self.frame_resizing = 0.25

        # Define augmentation sequence
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flips
            iaa.Affine(rotate=(-20, 20))  # Rotations
        ])

    def augment_image(self, image):
        """
        Apply augmentation to the given image.
        :param image: Input image to be augmented.
        :return: Augmented image.
        """
        return self.augmenter.augment_image(image)

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path.
        :param images_path: Path to the folder containing encoding images.
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print("Error loading image:", img_path)
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Augment the image
            augmented_img = self.augment_image(rgb_img)

            # Get the filename only from the initial file path
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Get encoding
            face_encodings = face_recognition.face_encodings(augmented_img)
            if not face_encodings:
                print("No face found in image:", img_path)
                continue

            img_encoding = face_encodings[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

        print("Encoding images loaded")

    def learn_new_face(self, image_path, name):
        """
        Learn a new face and update the known faces.
        :param image_path: Path to the image containing the new face.
        :param name: Name associated with the new face.
        """
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the encoding of the new face
        new_face_encoding = face_recognition.face_encodings(rgb_img)[0]

        # Update known faces
        self.known_face_encodings.append(new_face_encoding)
        self.known_face_names.append(name)

        # Save the updated known faces to a file
        self.save_known_faces()

    def save_known_faces(self):
        """
        Save known faces to a file.
        """
        data = {'encodings': self.known_face_encodings, 'names': self.known_face_names}
        with open('known_faces.pkl', 'wb') as file:
            pickle.dump(data, file)

    def load_known_faces(self):
        """
        Load known faces from a file, if available.
        """
        try:
            with open('known_faces.pkl', 'rb') as file:
                data = pickle.load(file)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print("Known faces loaded from file.")
        except FileNotFoundError:
            print("No previous known faces found.")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

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

# import cv2
# import mediapipe as mp
# import numpy as np
# import os

# # Initialize MediaPipe Pose.
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# # Input and output folder paths
# input_folder = 'DATASET'
# output_folder = 'skeletal_output'

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Process each image in the input folder
# for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         # Read the image
#         try:
#             image_path = os.path.join(input_folder, filename)
#             image = cv2.imread(image_path)
            
#             if image is None:
#                 print(f"Error reading image:{image_path}")
#                 continue
        
#             image_height, image_width, _ =image.shape

#             # Convert the image to RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             # Process the image and find the pose
#             result = pose.process(image_rgb)

#             # Create a blank numpy array with zeros
#             keypoints_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        

#             # Draw the keypoints on the blank image
#             if result.pose_landmarks:
#                 for landmark in result.pose_landmarks.landmark:
#                     x = int(landmark.x * image_width)
#                     y = int(landmark.y * image_height)
#                     cv2.circle(keypoints_image, (x, y), 5, (255, 255, 255), -1)

#                 # Save the image with keypoints
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, keypoints_image)
#                 print(f"Processed and saved: {filename}")
#             else:
#                 print(f"No pose detected!")

#         except Exception as e:
#             print(f"Error processing image {filename}: {str(e)}")

# print("Processing complete.")


# import cv2
# import mediapipe as mp
# import numpy as np
# import os

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# def process_images(input_path, output_path):
#     # Create output folder if it doesn't exist
#     os.makedirs(output_path, exist_ok=True)
    
#     # Process each image in the input folder
#     for class_folder in os.listdir(input_path):
#         class_path = os.path.join(input_path, class_folder)
#         if os.path.isdir(class_path):
#             # Create corresponding output class folder
#             output_class_path = os.path.join(output_path, class_folder)
#             os.makedirs(output_class_path, exist_ok=True)
            
#             for filename in os.listdir(class_path):
#                 if filename.endswith(('.png', '.jpg', '.jpeg')):
#                     try:
#                         # Read the image
#                         image_path = os.path.join(class_path, filename)
#                         image = cv2.imread(image_path)
                        
#                         if image is None:
#                             print(f"Error reading image: {image_path}")
#                             continue
                        
#                         image_height, image_width, _ = image.shape
                        
#                         # Convert the image to RGB
#                         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
#                         # Process the image and find the pose
#                         result = pose.process(image_rgb)
                        
#                         # Create a blank numpy array with zeros
#                         keypoints_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                        
#                         # Draw the keypoints on the blank image
#                         if result.pose_landmarks:
#                             for landmark in result.pose_landmarks.landmark:
#                                 x = int(landmark.x * image_width)
#                                 y = int(landmark.y * image_height)
#                                 cv2.circle(keypoints_image, (x, y), 5, (255, 255, 255), -1)
                            
#                             # Save the image with keypoints
#                             output_path_file = os.path.join(output_class_path, filename)
#                             cv2.imwrite(output_path_file, keypoints_image)
#                             print(f"Processed and saved: {class_folder}/{filename}")
#                         else:
#                             print(f"No pose detected in: {class_folder}/{filename}")
                            
#                     except Exception as e:
#                         print(f"Error processing {class_folder}/{filename}: {str(e)}")

# # Process both TRAIN and TEST directories
# base_input = 'DATASET'
# base_output = 'skeletal_output'

# # Process training data
# train_input = os.path.join(base_input, 'TRAIN')
# train_output = os.path.join(base_output, 'TRAIN')
# print("Processing training images...")
# process_images(train_input, train_output)

# # Process test data
# test_input = os.path.join(base_input, 'TEST')
# test_output = os.path.join(base_output, 'TEST')
# print("Processing test images...")
# process_images(test_input, test_output)

# print("Processing complete.")

import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class YogaPoseClassifier:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load the trained model
        self.model = load_model('best_model.h5')
        
        # Get class names from your training data
        self.class_names = sorted(os.listdir('DATASET/TRAIN'))
        
    def process_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find pose
        results = self.pose.process(frame_rgb)
        
        # Create blank image for skeleton
        h, w, _ = frame.shape
        skeleton_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                skeleton_image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Prepare image for classification
            skeleton_resized = cv2.resize(skeleton_image, (75, 75))
            skeleton_array = img_to_array(skeleton_resized)
            skeleton_array = skeleton_array / 255.0
            skeleton_array = np.expand_dims(skeleton_array, axis=0)
            
            # Predict pose
            predictions = self.model.predict(skeleton_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            pose_name = self.class_names[predicted_class]
            
            return skeleton_image, True, pose_name, confidence
            
        return skeleton_image, False, None, None

    def run_webcam(self):
        # Create output directory
        output_dir = 'webcam_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam started. Press:")
        print("'s' to save current skeleton")
        print("'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            skeleton_image, pose_detected, pose_name, confidence = self.process_frame(frame)
            
            # Add text to display pose and confidence
            if pose_detected and pose_name and confidence:
                text = f"{pose_name}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show original and skeleton images side by side
            combined_image = np.hstack((frame, skeleton_image))
            cv2.imshow('Yoga Pose Classification', combined_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and pose_detected:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(output_dir, f'skeleton_{timestamp}.jpg')
                cv2.imwrite(save_path, skeleton_image)
                print(f"Saved skeleton image to: {save_path}")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

if __name__ == "__main__":
    classifier = YogaPoseClassifier()
    classifier.run_webcam()
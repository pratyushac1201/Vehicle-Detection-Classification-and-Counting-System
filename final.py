import os
from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, '3.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

class_name_dict = {0: 'Two wheeler(Bikes,Scooter,Cycle)', 1: 'Auto(Three Wheeler)', 2: 'Four Wheeler(Cars,Mini-Trucks,Taxi)',
                   3: 'Large Vehicles(Lorry,Trucks,Tanker)'}

# Initialize a dictionary to keep track of counts
class_counts = {class_id: 0 for class_id in class_name_dict.keys()}
active_vehicles = []

# Define an ROI (x1, y1, x2, y2)
roi = (100, 100, W-100, H-100)  # Example ROI

vehicle_id = 0  # Unique identifier for each vehicle

while ret:
    results = model(frame)[0]

    current_detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            # Check if detection is within the ROI
            if x1 >= roi[0] and y1 >= roi[1] and x2 <= roi[2] and y2 <= roi[3]:
                # Check if this detection matches any active vehicle
                match_found = False
                for vehicle in active_vehicles:
                    if vehicle['class_id'] == class_id and np.linalg.norm(np.array([x1, y1]) - np.array([vehicle['x1'], vehicle['y1']])) < 50:
                        # Update vehicle position
                        vehicle['x1'], vehicle['y1'], vehicle['x2'], vehicle['y2'] = x1, y1, x2, y2
                        match_found = True
                        break

                if not match_found:
                    # New vehicle detected within the ROI
                    class_counts[int(class_id)] += 1
                    active_vehicles.append({'id': vehicle_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class_id': class_id})
                    vehicle_id += 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, f"{class_name_dict[int(class_id)].upper()} ID:{vehicle_id}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Draw the ROI for visualization
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)

    # Display counts on the frame
    startY = 30
    for class_id, count in class_counts.items():
        text = "{}: {}".format(class_name_dict[class_id], count)
        cv2.putText(frame, text, (10, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        startY += 30

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Break the loop if 'q' key is pressed
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

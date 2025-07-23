import os
import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')

def process_video(video_path, roi, live=False):
    # Parse ROI coordinates
    x1, y1, x2, y2 = map(int, roi.split(','))

    if live:
        cap = cv2.VideoCapture(0)  # 0 for default camera
    else:
        cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()     
    if not ret:
        st.error("Failed to read video")
        return
    
    H, W, _ = frame.shape
    out_path = video_path.replace('.mp4', '_out.mp4') if not live else 'live_output.mp4'
    # Change codec to 'avc1' for better browser compatibility
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)
    threshold = 0.5

    class_name_dict = {0: 'Two wheeler(Bikes,Scooter,Cycle)', 1: 'Auto(Three Wheeler)', 2: 'Four Wheeler(Cars,Mini-Trucks,Taxi)',
                       3: 'Large Vehicles(Lorry,Trucks,Tanker)'}
    class_counts = {class_id: 0 for class_id in class_name_dict.keys()}
    active_vehicles = []
    vehicle_id = 0

    with st.spinner('Processing video...'):
        while ret:
            results = model(frame)[0]
            for result in results.boxes.data.tolist():
                x1_box, y1_box, x2_box, y2_box, score, class_id = result
                if score > threshold and x1_box >= x1 and y1_box >= y1 and x2_box <= x2 and y2_box <= y2:
                    match_found = False
                    for vehicle in active_vehicles:
                        if vehicle['class_id'] == class_id and np.linalg.norm(np.array([x1_box, y1_box]) - np.array([vehicle['x1'], vehicle['y1']])) < 50:
                            vehicle['x1'], vehicle['y1'], vehicle['x2'], vehicle['y2'] = x1_box, y1_box, x2_box, y2_box
                            match_found = True
                            break

                    if not match_found:
                        class_counts[int(class_id)] += 1
                        active_vehicles.append({'id': vehicle_id, 'x1': x1_box, 'y1': y1_box, 'x2': x2_box, 'y2': y2_box, 'class_id': class_id})
                        vehicle_id += 1

                    cv2.rectangle(frame, (int(x1_box), int(y1_box)), (int(x2_box), int(y2_box)), (0, 255, 0), 4)
                    #cv2.putText(frame, f"{class_name_dict[int(class_id)].upper()} ID:{vehicle_id}", (int(x1_box), int(y1_box - 10)),
                                #cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Display counts on the frame
            startY = 30
            for class_id, count in class_counts.items():
                text = "{}: {}".format(class_name_dict[class_id], count)
                cv2.putText(frame, text, (10, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                startY += 30

            out.write(frame)
            if live:
                stframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(stframe)
            ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return out_path if not live else None

def st_interface(video, roi, live=False):
    processed_video_path = process_video(video, roi, live)
    return processed_video_path

st.title('Vehicle Detection & Counting System')
video_file = st.file_uploader("Upload Video", type=['mp4'])
roi_input = st.text_input("ROI (x1,y1,x2,y2)", "0,0,10000,10000")
live_option = st.checkbox("Process from Live Camera")

if st.button("Process Video"):
    if live_option:
        st_interface(None, roi_input, live=True)
    elif video_file is not None and roi_input:
        video_path = os.path.join(VIDEOS_DIR, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        processed_video = st_interface(video_path, roi_input)
        st.video(processed_video)
                                    
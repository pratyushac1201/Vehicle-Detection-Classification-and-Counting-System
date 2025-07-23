import streamlit as st
import os
from ultralytics import YOLO
import cv2
import tempfile

# Function to perform YOLOv8 detection on a video
def perform_yolov8_detection(video_path, yolov8_weights_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out_path = os.path.join('.', 'output_video.mp4')  # Save the output video in the project directory
    out = cv2.VideoWriter(out_path, -1, int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model = YOLO(yolov8_weights_path)

    threshold = 0.5

    class_name_dict = {0: 'Two wheeler(Bikes,Scooter,Cycle)', 1: 'Auto(Three Wheeler)', 2: 'Four Wheeler(Cars,Mini-Trucks,Taxi)',
                       3: 'Large Vehicles(Lorry,Trucks,Tanker)'}

    gif_path = "loading.gif"
    loading_gif = st.image(gif_path, use_column_width=True)  # Display the loading GIF

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Convert the frame to RGB format for displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Remove loading GIF and display final output video
    loading_gif.empty()  # Remove the loading GIF from the Streamlit app
    st.title("YOLOv8 Detection Result")
   

# Streamlit app
def main():
    st.title("YOLOv8 Video Testing")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Display uploaded video
        video_file = tempfile.NamedTemporaryFile(delete=False)
        video_file.write(uploaded_file.read())
        st.video(video_file.name)
        


        # Perform YOLOv8 detection
        yolov8_weights_path = "runs/detect/train/weights/best.pt"  # Replace with the actual path
        
        perform_yolov8_detection(video_file.name, yolov8_weights_path)
        st.video("output_video.mp4")
       
       
if __name__ == "__main__":
    main()

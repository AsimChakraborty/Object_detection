# from flask import Flask, render_template, request, redirect
# import os
# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image

# # Initialize Flask app
# app = Flask(__name__)

# # Define upload and output folders
# UPLOAD_FOLDER = "static/uploads/"
# DETECTED_FOLDER = "static/detected/"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["DETECTED_FOLDER"] = DETECTED_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(DETECTED_FOLDER, exist_ok=True)

# # Load YOLOv8 model
# model = YOLO("best.pt")  # Ensure "best.pt" is trained with YOLOv8

# # Allowed file types
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image(image_path):
#     img = Image.open(image_path).convert("RGB")

#     # Run YOLOv8 model
#     results = model(img)

#     # Read original image for drawing
#     img_cv2 = cv2.imread(image_path)
#     h, w, _ = img_cv2.shape

#     # Initialize object counts
#     class_counts = {}

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             conf = float(box.conf[0])  # Confidence score
#             cls = int(box.cls[0])  # Class ID
#             class_name = model.names[cls]  # Class name

#             class_counts[class_name] = class_counts.get(class_name, 0) + 1

#             # Draw bounding box
#             color = (0, 255, 0)
#             cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img_cv2, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     output_path = os.path.join(DETECTED_FOLDER, os.path.basename(image_path))
#     cv2.imwrite(output_path, img_cv2)

#     return output_path, class_counts

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)

#         file = request.files["file"]

#         if file.filename == "" or not allowed_file(file.filename):
#             return redirect(request.url)

#         # Save uploaded file
#         filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(filepath)

#         # Process the uploaded image
#         detected_path, counts = process_image(filepath)

#         return render_template("index.html", image_path=detected_path, counts=counts)

#     return render_template("index.html", image_path=None, counts={})

# if __name__ == "__main__":
#     app.run(debug=True)








# from flask import Flask, render_template, request, redirect
# import os
# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image

# # Initialize Flask app
# app = Flask(__name__)

# # Define upload and output folders
# UPLOAD_FOLDER = "static/uploads/"
# DETECTED_FOLDER = "static/detected/"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["DETECTED_FOLDER"] = DETECTED_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(DETECTED_FOLDER, exist_ok=True)

# # Load YOLOv8 model
# model = YOLO("best.pt")  # Ensure "best.pt" is trained with YOLOv8

# # Allowed file types
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def process_image_or_video(file_path):
#     """Process image or video."""
#     # If file is a video
#     if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
#         video_capture = cv2.VideoCapture(file_path)
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
#         output_path = file_path.replace("uploads", "detected").replace(".mp4", "_output.mp4")
#         out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

#         # Read and process each frame from the video
#         while video_capture.isOpened():
#             ret, frame = video_capture.read()
#             if not ret:
#                 break

#             # Run YOLOv8 model on the frame
#             results = model(frame)

#             # Draw bounding boxes and labels on the frame
#             for result in results:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                     conf = float(box.conf[0])  # Confidence score
#                     cls = int(box.cls[0])  # Class ID
#                     class_name = model.names[cls]  # Class name

#                     # Draw bounding box and label
#                     color = (0, 255, 0)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # Write the processed frame to the output video
#             out.write(frame)

#         video_capture.release()
#         out.release()
#         return output_path, None  # No object counts for videos

#     else:
#         # If file is an image, process it using PIL
#         img = Image.open(file_path).convert("RGB")

#         # Run YOLOv8 model
#         results = model(img)

#         # Read the original image for drawing
#         img_cv2 = cv2.imread(file_path)
#         h, w, _ = img_cv2.shape

#         # Initialize object counts
#         class_counts = {}

#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                 conf = float(box.conf[0])  # Confidence score
#                 cls = int(box.cls[0])  # Class ID
#                 class_name = model.names[cls]  # Class name

#                 class_counts[class_name] = class_counts.get(class_name, 0) + 1

#                 # Draw bounding box
#                 color = (0, 255, 0)
#                 cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(img_cv2, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         output_path = os.path.join(DETECTED_FOLDER, os.path.basename(file_path))
#         cv2.imwrite(output_path, img_cv2)

#         return output_path, class_counts

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)

#         file = request.files["file"]

#         if file.filename == "" or not allowed_file(file.filename):
#             return redirect(request.url)

#         # Save uploaded file
#         filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(filepath)

#         # Process the uploaded image or video
#         detected_path, counts = process_image_or_video(filepath)

#         return render_template("index.html", image_path=detected_path, counts=counts)

#     return render_template("index.html", image_path=None, counts={})

# if __name__ == "__main__":
#     app.run(debug=True)







from flask import Flask, render_template, request, redirect
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Define upload and output folders
UPLOAD_FOLDER = "static/uploads/"
DETECTED_FOLDER = "static/detected/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DETECTED_FOLDER"] = DETECTED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure "best.pt" is trained with YOLOv8

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_or_video(file_path):
    """Process image or video."""
    # If file is a video
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        video_capture = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
        output_path = file_path.replace("uploads", "detected").replace(".mp4", "_output.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

        # Read and process each frame from the video
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Run YOLOv8 model on the frame
            results = model(frame)

            # Draw bounding boxes and labels on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = float(box.conf[0])  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    class_name = model.names[cls]  # Class name

                    # Draw bounding box and label
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Write the processed frame to the output video
            out.write(frame)

        video_capture.release()
        out.release()
        return output_path, {}  # Return empty dict for video counts

    else:
        # If file is an image, process it using PIL
        img = Image.open(file_path).convert("RGB")

        # Run YOLOv8 model
        results = model(img)

        # Read the original image for drawing
        img_cv2 = cv2.imread(file_path)
        h, w, _ = img_cv2.shape

        # Initialize object counts
        class_counts = {}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class ID
                class_name = model.names[cls]  # Class name

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_cv2, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = os.path.join(DETECTED_FOLDER, os.path.basename(file_path))
        cv2.imwrite(output_path, img_cv2)

        return output_path, class_counts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Process the uploaded image or video
        detected_path, counts = process_image_or_video(filepath)

        return render_template("index.html", image_path=detected_path, counts=counts)

    return render_template("index.html", image_path=None, counts={})

if __name__ == "__main__":
    app.run(debug=True)

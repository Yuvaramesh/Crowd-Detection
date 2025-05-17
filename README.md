# 👥 Crowd Detection using YOLOv8

This Streamlit web application detects crowded areas in a video using the YOLOv8 object detection model. It analyzes frames of a video to detect people and clusters them based on their proximity to highlight potential crowd formations.

---

## 🚀 Features

- 🔍 Detects persons in uploaded video using **YOLOv8** (`yolov8n.pt`)
- 📏 Calculates distances between detected persons
- 🔵 Identifies crowd groups based on customizable distance and size thresholds
- 🎥 Displays video frames every 10th frame with bounding boxes
- 📊 Outputs crowd count per analyzed frame in a table
- 📁 Allows users to download the results as a CSV file

---

## 📸 Screenshots
### UI using Streamlit

![image](https://github.com/user-attachments/assets/3389c3b4-5404-4b0b-ae70-ca7025d4a2d0)

### Group of people detection through video

![image](https://github.com/user-attachments/assets/8aef1b88-d5be-497c-931e-47182c529f77)

### Output with frames and count of people

![image](https://github.com/user-attachments/assets/8f2f74a7-0aab-4fc4-8620-a5e987a62725)

### Output in CSV file

![image](https://github.com/user-attachments/assets/ff00f0cd-7f6d-4829-9b09-e6c4fcadc906)

---

## 🧠 Model

This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model for real-time object detection, focusing on the "person" class (class ID: 0).

---

## ⚙️ Requirements

Install the required dependencies using pip:

```bash
pip install streamlit opencv-python pandas ultralytics scipy pillow
````

You may also need to install `ffmpeg` depending on your platform for video decoding.

---

## 🛠️ How to Use

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/crowd-detection-yolov8.git
   cd crowd-detection-yolov8
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Upload a video file (mp4, avi, or mov).

4. Adjust the **Distance Threshold** and **Minimum Crowd Size** sliders to define what constitutes a crowd.

5. Watch the analysis live and download the results as a CSV file.

---

## 📁 Output

* **Live frame display**: Bounding boxes in green (crowd) and blue (not in crowd)
* **CSV Report**: Every 10th frame analyzed with crowd count:

  ```csv
  Frame Number,Person Count in Crowd
  10,5
  20,3
  ...
  ```

---

## 📌 Parameters

| Parameter          | Description                                         |
| ------------------ | --------------------------------------------------- |
| Distance Threshold | Max distance (in pixels) between persons in a group |
| Minimum Crowd Size | Minimum number of people considered as a crowd      |

---

## 📂 File Structure

```bash
crowd-detection-yolov8/
│
├── app.py                   # Main Streamlit application
├── requirements.txt         # Dependencies (optional)
└── README.md                # This file
```

---

## ✅ Future Enhancements

* Add heatmap overlays for visualizing dense zones
* Process entire video and output annotated video file
* Support live webcam crowd detection
* Add support for alert generation when crowds exceed a threshold

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)
* [Scipy](https://www.scipy.org/)

---

# Demo Link 
[Do Check it Out!](https://crowd-detection-by-yuva.streamlit.app/)
# 🧑‍💻 Author<br>
Yuva Sri Ramesh
[Portfolio](https://yuva-sri-ramesh-portfolio.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/yuvasri-r/) | [GitHub](https://github.com/Yuvaramesh)

📜 License
This project is licensed under the MIT License.

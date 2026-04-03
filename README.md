# SilentGuardian

# About the Project

SilentGurdian is a simple, smart safety system designed to support elderly and vulnerable individuals. It continuously monitors for critical events that could lead to injury or serious complications.
Using AI, the system can accurately detect falls in real time and recognize specific hand gestures used to signal for help. This dual detection approach enhances reliability and ensures timely responses during emergencies. 
The system is depolyed on a low-cost, efficient device such as the Raspberry Pi 5, making it accessible, and practical for real-world use.

### Features
- AI Detection
- Fall detection using trained models
- Hand gesture recognition for SOS/help signals
- Lightweight and optimized for devices like Raspberry Pi

### Flask Backend
- User signup, login, logout
- OTP-based password recovery for account security

### Frontend Integration
- Shows alerts when danger or emergency gestures are detected
- Works with mobile or simple web-based UI

### Raspberry Pi Compatible
- Supports Pi Camera
- Can run inference locally or stream to backend

### To run application:
- run "python app.py"  or run "flask run" from the application folder in the terminal

### Installation
1. Clone the Project
    - git clone repo--url
    - cd SilentGuardian
  
2. Install Dependencies
    - pip install -r requirements.txt

How to Run the Application
- Run the Backend
- From the project folder:
 python app.py
 or
 flask run
The app will be available at:
http://localhost:5000

# Running the Gesture Detection System on the Raspberry Pi

1. Run the script app.py
2. Connect the Raspberry Pi to the mobile PC hotspot
    - This shows the device name and the ip address
3. Enable SSH into the PC
4. Activate the virtual environment 
    - source gesture-env/bin/activate
    - cd fall_project
5. Run the script using this format 
    - python3 realtime_communication.py --email <"email_address"> --location "name_of_location" --api_base http://<"ip_address">:5000/
    - To do a live stream use http://<"ip_address">:8000/video_feed

# Contributors & Support 

- Deborah Ajayi (Team Member)
- Anique Ali (Team Member)
- Kamsiyochukwu Ekweozor (Team Member)
- Jennifer Ogidi-Gbegbaje (Team Member)
- Dr Safaa Bedawi (Supervisor)
- Dr Rami Sabouni (Co-supervisor)
- SYSC 4907 (Course)
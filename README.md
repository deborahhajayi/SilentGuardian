# SilentGuardian
SilentGurdian is a simple, smart safety system that quietly looks out for the people you care about most. Using AI, it can recognize when someone has fallen or when they make a hand gesture asking for help.

To run application:

1. Open the new terminal and activate the environemnt then on that terminal run the app.py using the command python app.py or you may also use flask run to run the client side. Login to the sysem with email and password.

2. Now that the client is running run the server on new terminal run the server in this format python src\realtime.py --email the_email_you_will_use_to_login_to_the_system_without_quotation --location "location_where_camera_is_displayed"

3. The camera will now be open and the fall will be detected. 














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




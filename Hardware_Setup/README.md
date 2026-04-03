## Raspberry Pi Setup Guide

### Overview 
This guide provides step-by-step instructions to configure a Raspberry Pi for running the system. It includes installing dependencies, setting up the environment, and enabling camera and headless operation. 

---

## System Setup

### Step 0: Update the System 

```bash
sudo apt update
sudo apt upgrade -y 
```

### Step 1: Install System Dependencies 

```bash
sudo apt install -y \
  python3-pip \
  python3-venv \
  python3-dev \
  libatlas-base-dev \
  libjpeg-dev \
  libpng-dev \
  libopenjp2-7 \
  libtiff6 \
  libglib2.0-0 \
  libstdc++6
```

### Step 2: Create a virtual environment 

```bash 
python3 -m venv --system-site-packages <virtual_env_name>
source <virtual_env_name>/bin/activate
```

### Step 3: Upgrade pip & Tooling

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Numpy 

```bash
pip install "numpy>=1.24.0"
```

### Step 5: Install OpenCV (Headless)

```bash
pip install "opencv-python-headless>=4.8.0"
```

### Step 6: Install TensorFlow Lite Runtime

```bash
pip install tflite-runtime
```

### Step 7: Install Picamera2 (Global Installation)

```bash
pip install picamera2
```

### Step 8: Install Flask

```bash
pip install Flask
```

## Camera Configuration

The IMX708 camera requires additional configuration. Follow the [Arducam IMX708 Camera Documentation](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/12MP-IMX708/#2-camera-usage) for detailed steps. 


File Transfer gdown (Global)

### Step 1: Install gdown (Global)

```bash
python3 -m pip install --user gdown
```

### Step 2: Download Shared Files

```bash
gdown <LINK_SHARING_ID> -O <script_name>
```

### Step 3: Verify File Type

```bash
file <script_name>
```

## Running Raspberry Pi in Headless Mode

- To run the Raspberry Pi without a monitor, SSH must be enabled.

### Step 1: Configure Boot Files

- Open the boot partition
- Create the following files:
    - ssh
    - wpa_supplicant.conf

Paste the following into wpa_supplicant.conf:

```bash
country=CA
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
  ssid="YOUR_HOTSPOT_NAME"
  psk="YOUR_HOTSPOT_PASSWORD"
  key_mgmt=WPA_PSK
}
```

- Note: If using different credentials, key_mgmt may need to be removed.

### Step 2: Boot the Rasbperry Pi

- Insert the SD card
- Ensure the hotspot is ON
- Power on the Raspberry Pi
- The device should automatically connect

### Step 3: SSH from Windows

```bash
ssh <raspberrypi_username>@<ip_address>
```

Important Notes

- Double check the connection of the Raspberry Pi ribbon cable. Ensure that it is fitted properly.
- Use a stable connection of up to 2.4GHz or preferably 5GHz


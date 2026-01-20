from flask import Flask, redirect, render_template, request, flash,session
import sqlite3
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
from datetime import datetime
from flask_cors import CORS
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app) 

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/signUp')
def signUp():

    return render_template('signUp.html')

@app.route('/home')
def home():
    fname = request.args.get('fname')
    lname = request.args.get('lname')
    email = request.args.get('email')

    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()
    falls = cursor.execute(
        "SELECT timestamp, location, image_path FROM FALL_EVENTS WHERE email=? ORDER BY timestamp DESC LIMIT 10",
        (email,)
    ).fetchall()
    connection.close()

    return render_template(
        'home.html',
        fname=fname,
        lname=lname,
        email=email,
        falls=falls
    )
@app.route('/login_validation' ,methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()

    user = cursor.execute("SELECT * FROM USERS WHERE email=? and password=?",(email,password)).fetchall()
    connection.close()

    if(len(user)>0):
        return redirect(f'/home?fname={user[0][0]}&lname={user[0][1]}&email={user[0][2]}')
    else:
        return redirect('/login')
    
@app.route('/add_user', methods=['POST'])
def add_user():
    fname = request.form.get('fname')
    lname = request.form.get('lname')
    email = request.form.get('email')
    password = request.form.get('password')

    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()

    ans = cursor.execute("SELECT * from USERS where email=? AND password=?",(email,password)).fetchall()
    if(len(ans)>0):
        connection.close()
        return render_template('signUp.html',msg="user already exists")
    else:
        cursor.execute("INSERT INTO USERS(first_name,last_name,email,password)values(?,?,?,?)",(fname,lname,email,password))
        connection.commit()
        connection.close()
        return render_template('login.html')

def generate_otp():
    """Generate a 6-digit OTP."""
    otp = random.randint(100000, 999999)
    return otp

def send_email(sender_email, receiver_email, subject, body, smtp_server, smtp_port, login, password):
    """Send an email with the specified parameters."""
    # Create the email message object
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach the HTML email body
    msg.attach(MIMEText(body, 'html'))

    try:
        # Establish connection to the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade to secure connection
        server.login(login, password)
        
        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully")
        server.quit()
        
    except Exception as e:
        print(f"Failed to send email: {e}")
    

@app.route('/forgot_page')
def forgot_page():
    return render_template('forgot-password.html')

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    global logged_mail
    logged_mail = ""
    email = request.form.get('email')
    logged_mail = email
    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()

    cmd1 = cursor.execute("SELECT * FROM USERS WHERE email=?", (email,)).fetchall()
    if len(cmd1) > 0:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        
        sender_email = "deborahajayi302@gmail.com"
        login = sender_email 
        
        password = "tftg jjqm ejee pkdc"  # Your application-specific password
        receiver_email = email
        subject = "Your OTP Code"
        otp = generate_otp()
        cursor.execute("DELETE FROM USEROTP WHERE email = ?", (email,))
        connection.commit()
        cursor.execute("INSERT INTO USEROTP(email, otp) VALUES (?, ?)", (email, otp))
        connection.commit()
        connection.close()

        body = f"""
        <html>
        <head>
        <style>
            .box {{
            width: 600px;
            background-color: #83c9c5;
            padding: 20px;
            box-shadow: 10px 15px 10px black;
            border-radius: 10px;
            height: 100px;
            align-items: center;
            justify-content: center;
        }}
        .box p {{
            font-size: 1rem;
        }}
        .box strong {{
            font-size: 1.2rem;
            margin-left: 3px;
            margin-right: 3px;
            background: orange;
            color: black;
            padding: 5px;
            border-radius: 5px;
            letter-spacing: 0.5px;
            cursor: pointer;
            transition: 1.2s linear ease;
        }}
        .box strong:hover{{
            transform: scale(1.2);
        }}
        </style>
        </head>
        <body>
            <div class="box">
                <p>Your OTP code is <strong>{otp}</strong>.</p>
            </div>
        </body>
        </html>
        """
        send_email(sender_email, receiver_email, subject, body, smtp_server, smtp_port, login, password)
        flash("otp has been sent!")
        return render_template('forgot-password.html',sent="OTP has been sent!")
    else:
        msg = "Invalid email!"
        return render_template('forgot-password.html', sent=msg)

@app.route('/check_otp', methods=['POST'])
def check_otp():
    otp = request.form.get('otp')
    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()
    #logged_mail = session.get('logged_mail') 
    if logged_mail:
        ans = cursor.execute("SELECT * FROM USEROTP WHERE email=?", (logged_mail,)).fetchall()
        if len(ans) > 0 and otp == ans[0][1]:
            cursor.execute("DELETE FROM USEROTP WHERE email = ?", (logged_mail,))
            user = cursor.execute("SELECT * from USERS where email=?",(logged_mail,)).fetchall()
            connection.commit()
            connection.close()
            return redirect(f'/home?fname={user[0][0]}&lname={user[0][1]}&email={user[0][2]}')
        else:
            connection.close()
            return render_template('forgot-password.html', msg="Invalid OTP!")

@app.route('/logout')
def logout():
    # Clear session data (if you use session later)
    global logged_mail
    logged_mail = ""
    return redirect('/')

@app.route('/profile')
def profile():
    fname = request.args.get('fname')
    lname = request.args.get('lname')
    email = request.args.get('email')
    return render_template('profile.html', fname=fname, lname=lname, email=email)

ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@app.route('/api/report_fall', methods=['POST'])
def report_fall():

    print("HIT /api/report_fall")
    print("content_type:", request.content_type)
    print("files:", list(request.files.keys()))
    print("form:", request.form.to_dict())
    print("json:", request.get_json(silent=True))

    
    data = request.form if request.form else (request.get_json(silent=True) or {})

    email = data.get('email')
    location = data.get('location', None)

    if not email:
        return {"status": "error", "message": "email required"}, 400

    # Save screenshot if provided
    image_path = None
    img = request.files.get('image')
    if img and img.filename:
        os.makedirs(os.path.join(app.root_path, "static", "falls"), exist_ok=True)
        filename = f"{uuid4().hex}.jpg"
        image_path = f"falls/{filename}"  # relative to /static
        img.save(os.path.join(app.root_path, "static", image_path))

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    connection = sqlite3.connect('LoginData.db', timeout=10)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO FALL_EVENTS(email, timestamp, location, image_path) VALUES (?,?,?,?)",
        (email, ts, location, image_path)
    )
    connection.commit()
    connection.close()

    return {"status": "ok", "image_path": image_path}, 200

if __name__ == '__main__':
    app.run(debug=True)
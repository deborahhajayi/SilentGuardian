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
from flask_bcrypt import Bcrypt
from datetime import timedelta
from functools import wraps

app = Flask(__name__)
app.secret_key = "fall-detection-secret-key"
CORS(app) 
bcrypt = Bcrypt(app)

# session expiry after certain timeline
app.permanent_session_lifetime = timedelta(minutes=40)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/signUp')
def signUp():

    return render_template('signUp.html')

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "email" not in session:
            flash("Session expired. Please log in again.", "error")
            return redirect('/')

        return f(*args, **kwargs)
    return wrapper

@app.route('/home')
@login_required
def home():
    fname = session.get('fname')
    lname = session.get('lname')
    email = session.get('email')

    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()
    falls = cursor.execute(
        """
        SELECT id, timestamp, location, image_path, status
        FROM FALL_EVENTS
        WHERE email=?
        ORDER BY timestamp DESC
        LIMIT 15
        """,
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

@app.route("/mark_false_positive", methods=["POST"])
def mark_false_positive():
    event_id = request.form.get("event_id")
    conn = sqlite3.connect("LoginData.db")
    c = conn.cursor()
    c.execute("UPDATE FALL_EVENTS SET status='false_alarm' WHERE id=?", (event_id,))
    conn.commit()
    conn.close()
    return redirect(request.referrer or '/home')


    


@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        flash("Please enter both email and password.", "error")
        return redirect('/')

    conn = sqlite3.connect('LoginData.db')
    cursor = conn.cursor()

    user = cursor.execute(
        "SELECT first_name, last_name, email, password_hash FROM USERS WHERE email=?",
        (email,)
    ).fetchone()

    conn.close()

    # ❌ Email not found
    if user is None:
        flash("Email does not match any account. Please register first.", "error")
        return redirect('/signUp')

    stored_hash = user[3]  # hashed password from DB

    # ❌ Password incorrect
    if not bcrypt.check_password_hash(stored_hash, password):
        flash("Incorrect password. Please try again.", "error")
        return redirect('/')

    # ✅ Login success
    session.permanent = True
    session['email'] = user[2]
    session['fname'] = user[0]
    session['lname'] = user[1]

    return redirect('/home')


    
@app.route('/add_user', methods=['POST'])
def add_user():
    fname = request.form.get('fname')
    lname = request.form.get('lname')
    email = request.form.get('email')
    password = request.form.get('password')

    # ---- Validation ----
    if not password:
        flash("Password required")
        return redirect("/signUp")

    if not email:
        flash("Email required")
        return redirect("/signUp")

    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()

    # ---- Check if user already exists (email ONLY) ----
    existing_user = cursor.execute(
        "SELECT 1 FROM USERS WHERE email = ?",
        (email,)
    ).fetchone()

    if existing_user:
        connection.close()
        return render_template('signUp.html', msg="User already exists")

    # ---- Hash password ----
    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    # ---- Insert new user ----
    cursor.execute(
        """
        INSERT INTO USERS (first_name, last_name, email, password_hash)
        VALUES (?, ?, ?, ?)
        """,
        (fname, lname, email, pw_hash)
    )

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
            session.permanent = True
            session['email'] = user[0][2]
            session['fname'] = user[0][0]
            session['lname'] = user[0][1]
            return redirect('/home')

        else:
            connection.close()
            return render_template('forgot-password.html', msg="Invalid OTP!")

@app.route('/logout')
def logout():
    # Clear session data (if you use session later)
    session.clear()
    flash("You have been logged out.", "success")
    global logged_mail
    logged_mail = ""
    return redirect('/')



@app.route('/profile')
@login_required
def profile():
    fname = session.get('fname')
    lname = session.get('lname')
    email = session.get('email')

    conn = sqlite3.connect("LoginData.db")
    cur = conn.cursor()

    falls = cur.execute(
    """
    SELECT 
    timestamp,
    location,
    status,
    strftime('%Y-%m', timestamp) as month,
    strftime('%W', timestamp) as week
    FROM FALL_EVENTS
    WHERE email=?
    ORDER BY timestamp DESC
    """,
    (email,)
    ).fetchall()

    conn.close()

    return render_template("profile.html",
                           fname=fname,
                           lname=lname,
                           email=email,
                           falls=falls)



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

@app.route('/confirm_fall', methods=['POST'])
def confirm_fall():
    event_id = request.form.get('event_id')
    connection = sqlite3.connect('LoginData.db')
    cursor = connection.cursor()
    cursor.execute("UPDATE FALL_EVENTS SET status='confirmed' WHERE id=?", (event_id,))
    connection.commit()
    connection.close()
    return redirect(request.referrer or '/home')

@app.route('/api/falls_latest')
@login_required
def falls_latest():
    email = session['email']
    conn = sqlite3.connect('LoginData.db')
    cur = conn.cursor()
    rows = cur.execute(
        """SELECT id, timestamp, location, image_path, status
           FROM FALL_EVENTS
           WHERE email=?
           ORDER BY timestamp DESC
           LIMIT 15""",
        (email,)
    ).fetchall()
    conn.close()

    return {"falls": rows}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
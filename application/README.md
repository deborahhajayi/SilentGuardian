# Login System with SQLite Database

 This web application allows users to sign up, log in, and manage their sessions using a simple login authentication mechanism. The user data is stored in an **SQLite** database. This project demonstrates a basic implementation of user authentication using **Flask**, with features like form validation and redirection.

##  Features

-  **User Authentication**: Login and signup functionality with email and password.
- **SQLite Integration**: Stores user data like first name, last name, email, and password in an SQLite database.
-  **HTML Templates**: Uses `Jinja` templating engine to display dynamic content on the home page.
-  **Session Redirection**: Redirects users based on successful login or signup.
- **Security**: Uses Flask's `secret_key` to manage session security.

## Technology Stack

- **Flask**: Python web framework used for routing and handling requests.
- **SQLite**: A lightweight database to store user credentials.
- **HTML & CSS**: For the user interface and layout.


## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Flask-Login-System.git
    ```

2. Install the required dependencies:
    ```bash
    pip install Flask
    ```

3. Run the Flask application:
    ```bash
    python app.py
    ```

4. Visit `http://127.0.0.1:5000/` in your browser.

## User Flow

1. **Signup**: New users can create an account by providing their first name, last name, email, and password.
2. **Login**: Existing users can log in using their email and password.
3. **Home**: After successful login, users are redirected to the home page displaying their details.
4. **Re-login**: If login credentials are incorrect, the user is redirected back to the login page.


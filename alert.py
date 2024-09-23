# alert.py

import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = os.getenv("EMAIL_PORT")
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True") == "True"
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

def send_email(subject, body):
    if not all([EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD, EMAIL_RECIPIENT]):
        print("Email alert is not configured. Skipping email.")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_HOST_USER
    msg['To'] = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(EMAIL_HOST, int(EMAIL_PORT)) as server:
            if EMAIL_USE_TLS:
                server.starttls()
            server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            server.send_message(msg)
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

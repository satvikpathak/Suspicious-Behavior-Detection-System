import smtplib
from email.mime.text import MIMEText

def send_alert(behavior, frame_id):
    msg = MIMEText(f"Suspicious behavior: {behavior} at frame {frame_id}")
    msg['Subject'] = 'Suspicious Activity Alert'
    msg['From'] = 'your_email@gmail.com'
    msg['To'] = 'recipient@domain.com'
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email@gmail.com', 'your_app_password')  # Use app-specific password
            server.send_message(msg)
        print(f"Email alert sent for {behavior}")
    except Exception as e:
        print(f"Failed to send email: {e}")
    print(f"ALERT: {behavior} detected at frame {frame_id}")
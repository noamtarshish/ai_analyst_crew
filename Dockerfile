# שלב 1 - תמונת בסיס עם פייתון
FROM python:3.11-slim

# שלב 2 - הגדרת תיקיית העבודה
WORKDIR /app

# שלב 3 - התקנת דרישות
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# שלב 4 - העתקת כל קבצי הפרויקט
COPY . .

# שלב 5 - פתיחת פורט של Streamlit
EXPOSE 8501

# שלב 6 - הפעלת האפליקציה
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]

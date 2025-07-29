# AiAnalystCrew Crew – Docker Edition

This project runs entirely inside Docker, so you don't need to install Python or any dependencies on your machine.

## 📦 Requirements
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed (if not bundled with Docker Desktop)

---

## 🚀 Running the Project with Docker

### 1️⃣ Build and start the container
In the root folder of the project, run:
```bash
docker compose up --build
```
This will:
- Build the Docker image using the included `Dockerfile`
- Install all dependencies inside the container
- Start the application

---

### 2️⃣ Access the application
Once the container is running, open:
```
http://localhost:8501
```
If you want to access it from another device on the same network, replace `localhost` with your computer’s IP address:
```
http://192.168.x.x:8501
```
(Find your IP using `ipconfig` on Windows or `ifconfig` / `ip a` on Mac/Linux.)

---

### 3️⃣ Stopping the application
Press `CTRL + C` in the terminal where it’s running  
or run:
```bash
docker compose down
```

---

## 🛠 Customization
To modify project settings:
- Update `src/ai_analyst_crew/config/agents.yaml` – define agents
- Update `src/ai_analyst_crew/config/tasks.yaml` – define tasks
- Update `.env` – add your `OPENAI_API_KEY`

---

## 📄 Project Structure
```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                # Your environment variables (e.g., OPENAI_API_KEY)
├── streamlit_app.py    # Main Streamlit entry point
└── src/ai_analyst_crew # Source code for agents, tasks, configs
```

---

## 💡 Notes
- No Python installation is required on your local machine.
- All dependencies are installed inside the Docker container.
- To update the code, simply edit your files and re-run:
```bash
docker compose up --build
```
- If you want to access from outside your network, you will need to configure port forwarding on your router or use a tunneling service like ngrok.

---

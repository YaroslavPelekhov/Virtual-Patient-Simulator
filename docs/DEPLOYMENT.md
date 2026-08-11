# Deployment Guide

This guide describes a fresh deployment. It intentionally does not rely on the project's previous development VPS because that host is no longer a verified deployment target.

## Production Architecture

Recommended components:

```text
Internet
   |
HTTPS reverse proxy
   |-- /api/*  -> FastAPI on 127.0.0.1:8000
   |-- /*      -> static frontend

Telegram bot -> HTTPS API endpoint
FastAPI -> LLM provider
FastAPI -> local SQLite or managed database
```

For a public deployment, add authentication and authorization before exposing teacher endpoints, session records, exports, or benchmark controls.

## 1. Server Preparation

Create an unprivileged service account and install Python 3.10+, Git, Nginx, and a certificate-management tool appropriate for the operating system.

```bash
sudo useradd --create-home --shell /bin/bash virtualpatient
sudo mkdir -p /opt/virtual-patient
sudo chown virtualpatient:virtualpatient /opt/virtual-patient
```

Clone the repository as that user and create the backend environment:

```bash
sudo -u virtualpatient git clone \
  https://github.com/YaroslavPelekhov/Virtual-Patient-Simulator.git \
  /opt/virtual-patient/app

cd /opt/virtual-patient/app/backend
sudo -u virtualpatient python3 -m venv .venv
sudo -u virtualpatient .venv/bin/pip install -r requirements.txt
sudo -u virtualpatient cp .env.example .env
```

Store production credentials with restrictive permissions:

```bash
sudo chown virtualpatient:virtualpatient /opt/virtual-patient/app/backend/.env
sudo chmod 600 /opt/virtual-patient/app/backend/.env
```

## 2. Backend Service

Example systemd unit:

```ini
[Unit]
Description=Virtual Patient FastAPI backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=virtualpatient
Group=virtualpatient
WorkingDirectory=/opt/virtual-patient/app/backend
EnvironmentFile=/opt/virtual-patient/app/backend/.env
ExecStart=/opt/virtual-patient/app/backend/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

SQLite and in-process session state make a single worker the safe default for this prototype. A multi-worker deployment requires shared state and database concurrency testing.

## 3. Static Frontend and Reverse Proxy

Example Nginx server block:

```nginx
server {
    listen 80;
    server_name training.example.org;

    root /opt/virtual-patient/app/frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 2m;
    }
}
```

The frontend uses same-origin API requests outside local development, so the reverse proxy can serve the UI and `/api/` from one public origin.

Enable HTTPS before connecting the Telegram bot or collecting any session data.

## 4. Telegram Bot Service

Create `bot/.env` from the example and set the public HTTPS backend URL.

```env
TELEGRAM_BOT_TOKEN=your_token
BACKEND_URL=https://training.example.org
```

Run the bot as a second supervised service under the same unprivileged account. Do not place the token directly in the unit file or command line.

## 5. Required Hardening

The repository is a research prototype, not a hardened clinical platform. Before public exposure:

- add authentication and role-based authorization;
- restrict access to teacher endpoints and exports;
- replace permissive CORS with explicit trusted origins;
- rate-limit chat and benchmark endpoints;
- validate maximum message and session sizes;
- sanitize upstream error responses;
- place secrets in a managed secret store;
- encrypt backups and define retention limits;
- avoid collecting identifiable patient information;
- add monitoring without logging dialogue content by default;
- run dependency and secret scans;
- document incident response and credential rotation.

## 6. Verification

After deployment, verify from a trusted machine:

```bash
curl -fsS https://training.example.org/api/cases
curl -fsS https://training.example.org/docs >/dev/null
```

Then run a short synthetic session and confirm that:

- the web client reaches the API over HTTPS;
- no credential appears in browser or service logs;
- teacher-only data is inaccessible to student accounts;
- the Telegram bot uses the expected backend;
- restart behavior preserves the required session records;
- backup and restore have been tested.

## Legacy Deployments

Legacy development hosts are not supported deployment targets. Create a fresh installation from a pinned repository revision, verify every SSH host fingerprint through an independent channel, and never bypass host-key validation.

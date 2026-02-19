# WALLET AI -- Portfolio Optimizer

## Installation and Execution Guide

------------------------------------------------------------------------

## Table of Contents

1.  Prerequisites\
2.  Backend Setup\
3.  Frontend Setup\
4.  First Access\
5.  Troubleshooting\
6.  Support\
7.  Final Notes

------------------------------------------------------------------------

## 1. Prerequisites

Before starting, make sure you have installed:

-   **Python 3.10 or higher**\
    https://www.python.org/downloads/

-   **Node.js 18 or higher (includes npm)**\
    https://nodejs.org/

-   **PostgreSQL 14 or higher**\
    https://www.postgresql.org/download/

### Verify installations

``` bash
python --version
node --version
npm --version
psql --version
```

------------------------------------------------------------------------

## 2. Backend Setup

### Step 1: Extract the Project

Unzip the project file into a folder of your choice.

------------------------------------------------------------------------

### Step 2: Create the PostgreSQL Database

#### Linux / Mac

``` bash
psql -U postgres
```

#### Windows (PowerShell as Administrator)

``` bash
psql -U postgres
```

Inside `psql`, execute:

``` sql
CREATE DATABASE tcc;
\q
```

------------------------------------------------------------------------

### Step 3: Configure Environment Variables

Navigate to the `backend` folder:

``` bash
cd path/to/backend
```

Create a file named `.env` with the following content:

``` env
DATABASE_URL=postgresql://postgres:YOUR_POSTGRES_PASSWORD@localhost:5432/tcc
JWT_SECRET_KEY=RandomSecureKey@2024
APP_NAME=Portfolio Optimizer

ENABLE_PRICE_SCHEDULER=true
PRICE_SCHEDULER_HOUR=23
PRICE_SCHEDULER_MINUTE=0
```

⚠ Replace `YOUR_POSTGRES_PASSWORD` with your actual PostgreSQL password.

------------------------------------------------------------------------

### Step 4: Create Python Virtual Environment

#### Linux / Mac

``` bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell)

``` bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (CMD)

``` bash
python -m venv venv
venv\Scripts\activate.bat
```

------------------------------------------------------------------------

### Step 5: Install Python Dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

### Step 6: Populate Database with Assets

``` bash
python manage.py setup
```

This may take 10--30 minutes depending on your internet connection.

------------------------------------------------------------------------

### Step 7: Start the Backend Server

``` bash
python app.py
```

Backend runs at:\
http://localhost:5000

Keep this terminal open.

------------------------------------------------------------------------

## 3. Frontend Setup

### Step 1: Open a New Terminal

Do NOT close the backend terminal.

------------------------------------------------------------------------

### Step 2: Navigate to Frontend

``` bash
cd path/to/frontend
```

------------------------------------------------------------------------

### Step 3: Install Dependencies

``` bash
npm install
```

------------------------------------------------------------------------

### Step 4: Start Frontend

``` bash
npm run dev
```

Frontend runs at:\
http://localhost:5173

------------------------------------------------------------------------

## 4. First Access

1.  Open http://localhost:5173\
2.  Click **Sign Up**\
3.  Create your account\
4.  Go to **Portfolios** → **Add Portfolio**\
5.  Fill optimization parameters\
6.  Click **Optimize and Create**

Optimization may take up to 2 minutes.

------------------------------------------------------------------------

## 5. Troubleshooting

### Connection refused

-   Ensure PostgreSQL is running\
-   Verify password in `.env`\
-   Test:

``` bash
psql -U postgres -d tcc
```

### ModuleNotFoundError

-   Activate virtual environment\
-   Run:

``` bash
pip install -r requirements.txt
```

### Port 5000 in use

Change port in `app.py`:

``` python
app.run(debug=True, port=5001)
```

Update frontend:

``` env
VITE_API_URL=http://localhost:5001
```

------------------------------------------------------------------------

## 6. Support

For academic project questions, contact the project author via course
coordination.

------------------------------------------------------------------------

## 7. Final Notes

-   Database is created automatically on first run\
-   Historical data comes from Yahoo Finance\
-   Initial load may take up to 30 minutes\
-   Daily updates run at 23:00\
-   Use Ctrl+C to stop servers

------------------------------------------------------------------------

End of Instructions

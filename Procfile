# Run the Flask API
web: FLASK_APP=backend/app.py flask run --host=0.0.0.0 --port=${PORT:-5000}

# Run the Streamlit dashboard
ui: streamlit run backend/dashboard.py --server.port=${PORT:-8501}

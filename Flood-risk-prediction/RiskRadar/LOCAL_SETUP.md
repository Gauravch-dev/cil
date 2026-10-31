# Run the two-page RiskRadar dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open http://127.0.0.1:8501/. Rainfall opens by default. Flood accepts original annual rainfall, climate and land/soil conditions and calls the supplied saved predictor locally; it does not need a second API service. Use Typical/Drier/Wetter examples or edit measured area inputs. No river-gauge elevation or observation date is required. The saved model and source data are preserved. Runtime inference selects CPU without rewriting model files.

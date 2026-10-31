# Run the manual flood-risk application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open http://127.0.0.1:8501/. Rainfall opens by default. Use Flood for the manual Kurundwad assessment. Enter river elevations now, six hours earlier and 24 hours earlier; rainfall is optional. The interface does not fetch live readings or show model/evaluation panels. Historical experiments and the former multi-page interfaces remain preserved in source/history.

Frozen experiment: experiments/kurundwad_flood_risk_v1. Use `scripts/verify_city_flood.py` to reconstruct every source label and verify saved results. Training refuses to overwrite frozen selection. The historical archive was previously inspected, so the new scores are retrospective, not independent prospective confirmation.

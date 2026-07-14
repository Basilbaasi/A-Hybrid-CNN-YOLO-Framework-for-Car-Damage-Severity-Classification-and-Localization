Place your trained CNN and YOLO models here
Store the trained inference artifacts in this directory:

- car.h5  (CNN severity classifier)
- best.pt (YOLO damage detector)

These files are intentionally excluded from Git. Their paths are configured in
damage_api/configs/params.yaml. With both files present, run:

uvicorn damage_api.app.main:app --host 0.0.0.0 --port 8000 --reload

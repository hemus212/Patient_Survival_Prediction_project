import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score
from fastapi import Response
import prometheus_client as prom
from patient_survival_prediction_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from patient_survival_prediction_model.predict import make_prediction

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
#print(sys.path)
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api import api_router
from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


test_data = dataframe = pd.read_csv(Path(f"{DATASET_DIR}/heart_failure_clinical_records_dataset.csv")) 
r2_metric = prom.Gauge('patient_survival_score', 'R2 score for random 100 testsamples')
# Function for updating metrics
def update_metrics():
    test = test_data.sample(100)
    test_feat = test.drop('cnt', axis=1)
    test_cnt = test['cnt'].values
    test_pred = make_prediction(input_data=test_feat)['predictions']
    r2 = r2_score(test_cnt, test_pred).round(3)
    r2_metric.set(r2)
    
@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain",content = prom.generate_latest())

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

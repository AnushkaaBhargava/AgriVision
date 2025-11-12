from fastapi import FastAPI
from routes import crop_recommendation_app, disease_routes

app = FastAPI(
    title="AgriVision",
    description="Unified Crop Recommendation & Plant Disease Detection API",
    version="1.0.0"
)

# ✅ Route integrations
app.include_router(crop_recommendation_app.router, prefix="/crop", tags=["Crop Recommendation"])
app.include_router(disease_routes.router, prefix="/disease", tags=["Plant Disease Detection"])

# ✅ Health check route
@app.get("/")
def home():
    return {"message": "Welcome to AgriVision API! Use /docs to test the endpoints."}


# ✅ For running directly (useful for debugging)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

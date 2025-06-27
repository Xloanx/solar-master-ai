from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import energy_audit, battery_design, pv_design, inverter_sizing, optimization, chat

app = FastAPI(title="Solar Installation AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
# app.include_router(energy_audit.router, prefix="/audit", tags=["Audit"])
# app.include_router(battery_design.router, prefix="/battery", tags=["Battery"])
# app.include_router(pv_design.router, prefix="/pv", tags=["PV"])
# app.include_router(inverter_sizing.router, prefix="/inverter", tags=["Inverter"])
# app.include_router(optimization.router, prefix="/optimize", tags=["Optimization"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])


#root path for testing
@app.get("/")
def root():
    return {"message": "Solar Master API is running"}
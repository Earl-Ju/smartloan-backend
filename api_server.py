"""
SmartLoan 预测 API 服务
======================
基于 FastAPI，供前端 /api/evaluate 调用。

运行：
  pip3 install fastapi uvicorn --break-system-packages
  python3 api_server.py

默认端口：8788（Cloudflare Pages 本地习惯端口）
"""

import os, sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# 确保能 import predict.py（同目录）
sys.path.insert(0, os.path.dirname(__file__))
from predict import Predictor

app = FastAPI(title="SmartLoan API", version="1.0.0")

# ── CORS + Private Network Access（Safari 需要）─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safari Private Network Access：所有响应加此 header
@app.middleware("http")
async def add_pna_header(request: Request, call_next):
    # 处理 preflight
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

# ── 全局加载模型（启动时执行一次）────────────────────────────────
predictor: Optional[Predictor] = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = Predictor()
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")

# ── 请求体 Schema ────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    age:                     int   = 30
    job_type:                str   = "private"   # gov/state/private/self/none
    work_years:              float = 0
    monthly_income:          float = 0
    provident_fund_months:   int   = 0
    bank_balance:            float = 0
    has_property:            bool  = False
    property_value:          float = 0
    property_loan:           float = 0
    existing_monthly_payment: float = 0
    credit_card_used:        float = 0
    credit_card_limit:       float = 0
    inquiries_6m:            int   = 0
    inquiries_2y:            int   = 0
    overdue_count:           int   = 0
    overdue_90d:             int   = 0
    loan_amount:             float = 100000
    loan_term:               int   = 12
    is_salary_bank:          bool  = False
    has_existing_product:    bool  = False

# ── 接口 ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "models")
    files = os.listdir(models_dir) if os.path.exists(models_dir) else []
    return {
        "status": "ok",
        "models_loaded": predictor is not None,
        "model_count": len(predictor.models) if predictor else 0,
        "models_dir": models_dir,
        "files_in_models_dir": sorted(files),
    }

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")

    form = req.model_dump()
    try:
        result = predictor.predict_all(form)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result

# ── 单产品预测（可选，供调试）────────────────────────────────────
@app.post("/api/evaluate/{product_id}")
async def evaluate_one(product_id: str, req: EvaluateRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    form = req.model_dump()
    return predictor.predict_one(form, product_id)

# ── 启动 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8788, reload=True)

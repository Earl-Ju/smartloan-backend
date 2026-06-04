"""
SmartLoan 预测 API 服务
======================
基于 FastAPI，供前端 /api/evaluate 和 /api/parse 调用。

运行：
  pip3 install fastapi uvicorn anthropic python-multipart --break-system-packages
  python3 api_server.py

默认端口：8788
"""

import os, sys, json, base64
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# 确保能 import predict.py（同目录）
sys.path.insert(0, os.path.dirname(__file__))
from predict import Predictor

app = FastAPI(title="SmartLoan API", version="1.1.0")

# ── CORS + Private Network Access（Safari 需要）─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_pna_header(request: Request, call_next):
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

# ── 全局加载模型 ──────────────────────────────────────────────────
predictor: Optional[Predictor] = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = Predictor()
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")

# ── 文档解析 Prompts ──────────────────────────────────────────────
PARSE_PROMPTS = {
    "credit_report": """你是中国银行贷款审批专家，擅长解读央行个人信用报告（人行征信报告）。
从这份征信报告（PDF或截图）中精确提取以下字段，只返回JSON，不要解释。

重要规则：
1. 查询次数【只统计"机构查询记录"中查询原因为"贷款审批"或"信用卡审批"的记录】
   - 不计入：贷后管理、本人查询、担保资格审查、法人代表/高管资信审查
2. 报告时间作为"今天"，据此计算近6个月和近2年的范围
3. 信用卡只统计人民币账户，忽略美元/外币账户
4. existing_monthly_payment：如报告中有明确的月还款额就填，否则填null

{
  "inquiries_6m": 近6个月内"贷款审批"+"信用卡审批"查询次数合计（整数），
  "inquiries_2y": 近2年内"贷款审批"+"信用卡审批"查询次数合计（整数），
  "overdue_count": 近2年逾期还款总次数（整数；信用概要中"发生过逾期的账户数"不为"--"则查明细累计；若显示"--"则为0），
  "overdue_90d": 是否有90天以上严重逾期（true/false；概要"发生过90天以上逾期的账户数"不为"--"则为true），
  "existing_monthly_payment": 当前未结清贷款月还款总额（数字，单位元；若报告中无此数据则为null），
  "credit_card_used": 所有人民币信用卡已使用额度合计（数字，单位元；余额栏或已使用额度栏求和），
  "credit_card_limit": 所有人民币信用卡授信额度合计（数字，单位元）
}

如某字段确实无法从报告中提取，值设为null。只输出JSON，不含任何其他文字。""",

    "social_insurance": """你是中国社保公积金分析专家。
从这份社保/公积金记录截图中提取信息，只返回JSON，不要解释。

{
  "provident_fund_months": 公积金连续缴纳月数（整数，如显示"正常"则估算月数）,
  "monthly_income": 月薪估算（数字，单位元；如有公积金缴费基数，月薪≈缴费基数；如有社保缴费基数亦可参考）,
  "job_type": 单位性质，映射为以下值之一：gov（政府/机关/事业单位/公务员）、state（国企/央企/国有企业）、private（私企/民营企业）、self（个体/自雇）
}

如某字段找不到，值设为null。只输出JSON。""",

    "property": """你是中国不动产评估专家。
从这份不动产权证/房产信息截图中提取信息，只返回JSON，不要解释。

{
  "has_property": true,
  "property_value": 房产市场参考价值（数字，单位元；如证书上有评估价/成交价则使用；否则null）,
  "property_loan": 房产剩余抵押贷款金额（数字，单位元；如无抵押则为0）
}

如某字段找不到，值设为null。只输出JSON。""",

    "bank_statement": """你是中国银行流水分析专家，专注于工资代发识别。
从这份银行流水PDF或截图中提取信息，只返回JSON，不要解释。

{
  "monthly_income": 月均工资到账金额（数字，单位元；识别"工资"/"薪资"/"代发"等规律性收入的月均值）,
  "bank_balance": 账户最近余额（数字，单位元）,
  "is_salary_bank": 是否为代发工资账户（true或false；如有"工资"/"薪资"/"代发"字样则为true）
}

如某字段找不到，值设为null。只输出JSON。""",

    "insurance": """你是中国保险分析专家。
从这份保险保单截图中提取信息，只返回JSON，不要解释。

{
  "insurance_type": 保险类型（寿险/重疾险/年金险/财产险/其他之一）,
  "insurance_value": 保单现金价值或保额（数字，单位元）,
  "insurer": 保险公司名称（字符串）
}

如某字段找不到，值设为null。只输出JSON。""",
}

# ── 文档解析接口 ──────────────────────────────────────────────────
@app.post("/api/parse")
async def parse_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    client_api_key: Optional[str] = Form(None),
):
    api_key = client_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="请先在页面设置 Anthropic API Key")

    prompt = PARSE_PROMPTS.get(doc_type)
    if not prompt:
        raise HTTPException(status_code=400, detail=f"未知文档类型: {doc_type}")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件过大，请压缩后重试（上限20MB）")

    filename = (file.filename or "").lower()
    content_type = file.content_type or ""
    is_pdf = filename.endswith(".pdf") or "pdf" in content_type

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.standard_b64encode(content).decode("utf-8")

        if is_pdf:
            file_block = {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            }
        else:
            mt = content_type if content_type.startswith("image/") else "image/jpeg"
            file_block = {
                "type": "image",
                "source": {"type": "base64", "media_type": mt, "data": b64},
            }

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": [file_block, {"type": "text", "text": prompt}]}],
        )

        raw = response.content[0].text.strip()
        # 去掉 markdown 代码块
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        fields = json.loads(raw)
        # 过滤 null 值
        fields = {k: v for k, v in fields.items() if v is not None}
        return {"doc_type": doc_type, "fields": fields, "ok": True}

    except json.JSONDecodeError:
        return {"doc_type": doc_type, "fields": {}, "ok": False, "error": "解析结果格式异常"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 请求体 Schema ─────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    age:                     int   = 30
    job_type:                str   = "private"
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

# ── 评估接口 ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "models")
    files = os.listdir(models_dir) if os.path.exists(models_dir) else []
    return {
        "status": "ok",
        "models_loaded": predictor is not None,
        "model_count": len(predictor.models) if predictor else 0,
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
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

@app.post("/api/evaluate/{product_id}")
async def evaluate_one(product_id: str, req: EvaluateRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    form = req.model_dump()
    return predictor.predict_one(form, product_id)

# ── 启动 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8788, reload=True)

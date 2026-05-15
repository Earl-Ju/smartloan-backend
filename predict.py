"""
预测接口
========
供后端 /api/evaluate 调用。
输入：用户表单字段（dict）
输出：每个产品的通过概率 + 综合建议

用法：
  from predict import Predictor
  p = Predictor()
  results = p.predict_all(form_data)
"""

import json, os
import numpy as np

BASE     = os.path.dirname(__file__)
META_PATH = os.path.join(BASE, "models/meta.json")

class Predictor:
    def __init__(self):
        import joblib
        with open(META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]
        self.job_types     = self.meta["job_types"]

        # 加载所有模型
        self.models = {}
        for name, info in self.meta["models"].items():
            path = info["path"]
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        print(f"Predictor 加载完成：{len(self.models)} 个模型")

    def _extract(self, form: dict) -> np.ndarray:
        income   = max(form.get("monthly_income", 1), 1)
        cc_limit = form.get("credit_card_limit", 0)
        dti      = form.get("existing_monthly_payment", 0) / income
        cc_util  = (form.get("credit_card_used", 0) / cc_limit) if cc_limit > 0 else 0
        net_prop = form.get("property_value", 0) - form.get("property_loan", 0)
        loan_k   = max(form.get("loan_amount", 1), 1) / 1000

        base = [
            form.get("age", 30),
            form.get("work_years", 0),
            income,
            form.get("provident_fund_months", 0),
            form.get("bank_balance", 0),
            form.get("existing_monthly_payment", 0),
            form.get("credit_card_used", 0),
            cc_limit,
            form.get("inquiries_6m", 0),
            form.get("inquiries_2y", 0),
            form.get("overdue_count", 0),
            form.get("overdue_90d", 0),
            int(form.get("has_property", False)),
            form.get("property_value", 0),
            form.get("property_loan", 0),
            form.get("loan_amount", 100000),
            form.get("loan_term", 12),
            int(form.get("is_salary_bank", False)),
            int(form.get("has_existing_product", False)),
            dti,
            cc_util,
            net_prop,
            income / loan_k,
        ]
        job = form.get("job_type", "private")
        base += [1 if job == jt else 0 for jt in self.job_types]
        return np.array([base], dtype=np.float32)

    def predict_one(self, form: dict, product_id: str) -> dict:
        """预测单个产品的通过概率"""
        X = self._extract(form)
        model = self.models.get(product_id) or self.models.get("global")
        if model is None:
            return {"product": product_id, "pass_prob": 0.5, "confidence": "low"}

        prob = float(model.predict_proba(X)[0][1])  # P(pass)
        confidence = "high" if prob > 0.75 or prob < 0.25 else "medium" if prob > 0.6 or prob < 0.4 else "low"
        return {
            "product":    product_id,
            "pass_prob":  round(prob, 3),
            "pass_pct":   int(prob * 100),
            "confidence": confidence,
        }

    def predict_all(self, form: dict) -> dict:
        """预测所有产品，返回排序后的结果 + 拒绝原因分析"""
        # 产品列表（排除 global）
        products = [k for k in self.models if k != "global"]
        if not products:
            products = ["cmb_flash", "icbc_rong_e", "ccb_quick",
                        "pingan_new1", "spdb_puhui", "citic_huimin"]

        predictions = [self.predict_one(form, pid) for pid in products]
        predictions.sort(key=lambda x: -x["pass_prob"])

        # 拒绝原因分析
        reasons = self._analyze_reasons(form)

        # 最高分产品的特征重要性解释
        best = predictions[0] if predictions else {}

        return {
            "predictions": predictions,
            "top_product": best.get("product"),
            "top_pass_prob": best.get("pass_prob", 0),
            "reject_reasons": reasons,
            "improvement_tips": self._improvement_tips(form, reasons),
        }

    def _analyze_reasons(self, form: dict) -> list[str]:
        """基于规则分析主要拒绝风险"""
        reasons = []
        income = form.get("monthly_income", 0)

        if form.get("overdue_90d", 0) > 0:
            reasons.append("存在90天以上严重逾期记录")
        if form.get("overdue_count", 0) >= 3:
            reasons.append(f"近2年逾期{form['overdue_count']}次，超出多数银行容忍上限")
        if form.get("inquiries_6m", 0) >= 6:
            reasons.append(f"近6月征信查询{form['inquiries_6m']}次，触发银行风控")
        elif form.get("inquiries_6m", 0) >= 4:
            reasons.append(f"近6月征信查询{form['inquiries_6m']}次，偏多")
        if income > 0:
            dti = form.get("existing_monthly_payment", 0) / income
            if dti > 0.65:
                reasons.append(f"月负债率{dti:.0%}，超过多数银行65%上限")
            elif dti > 0.5:
                reasons.append(f"月负债率{dti:.0%}，偏高")
        if income < 4000:
            reasons.append(f"月收入{income}元，低于多数银行准入门槛")
        if form.get("credit_card_limit", 0) > 0:
            util = form.get("credit_card_used", 0) / form["credit_card_limit"]
            if util > 0.85:
                reasons.append(f"信用卡使用率{util:.0%}，过高影响评分")
        if form.get("work_years", 0) < 1:
            reasons.append("工龄不足1年，影响收入稳定性评分")
        return reasons

    def _improvement_tips(self, form: dict, reasons: list[str]) -> list[str]:
        """针对主要风险给出改善建议"""
        tips = []
        for r in reasons[:3]:
            if "逾期" in r and "90天" in r:
                tips.append("严重逾期记录影响较大，建议等待记录满5年后再申请，或先申请容忍度更高的产品")
            elif "逾期" in r:
                tips.append("尽快还清所有逾期欠款，保持6个月以上良好还款记录")
            elif "查询" in r:
                tips.append("停止一切新的贷款/信用卡申请，等待3-6个月让查询次数自然降低")
            elif "负债率" in r:
                tips.append("优先提前偿还小额贷款，将月负债率降至50%以下再申请")
            elif "收入" in r:
                tips.append("尝试补充收入证明（兼职、投资等），或选择门槛更低的产品如平安新一贷")
            elif "信用卡" in r:
                tips.append("将信用卡使用率降至70%以下，可临时提高授信额度或减少使用")
            elif "工龄" in r:
                tips.append("在当前单位再工作满1年后申请，或提供其他收入稳定性证明")
        if not tips:
            tips.append("当前资质良好，建议优先申请通过率最高的推荐产品")
        return tips


# ── 命令行测试 ────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = Predictor()

    # 测试案例
    test_cases = [
        {
            "name": "优质客户",
            "form": {
                "age": 32, "job_type": "state", "work_years": 5,
                "monthly_income": 15000, "provident_fund_months": 36,
                "bank_balance": 80000, "has_property": True,
                "property_value": 1500000, "property_loan": 600000,
                "existing_monthly_payment": 3000, "credit_card_used": 5000,
                "credit_card_limit": 30000, "inquiries_6m": 1,
                "inquiries_2y": 3, "overdue_count": 0, "overdue_90d": 0,
                "loan_amount": 100000, "loan_term": 24,
                "is_salary_bank": True, "has_existing_product": True,
            }
        },
        {
            "name": "高风险客户",
            "form": {
                "age": 28, "job_type": "private", "work_years": 1,
                "monthly_income": 5000, "provident_fund_months": 0,
                "bank_balance": 2000, "has_property": False,
                "property_value": 0, "property_loan": 0,
                "existing_monthly_payment": 3500, "credit_card_used": 18000,
                "credit_card_limit": 20000, "inquiries_6m": 7,
                "inquiries_2y": 12, "overdue_count": 2, "overdue_90d": 1,
                "loan_amount": 200000, "loan_term": 36,
                "is_salary_bank": False, "has_existing_product": False,
            }
        },
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试案例：{case['name']}")
        result = predictor.predict_all(case["form"])
        print(f"  TOP产品推荐:")
        for p in result["predictions"][:4]:
            bar = "█" * int(p["pass_pct"] / 5)
            print(f"    {p['product']:20s} {p['pass_pct']:3d}% {bar}")
        if result["reject_reasons"]:
            print(f"  主要风险:")
            for r in result["reject_reasons"]:
                print(f"    ⚠ {r}")
        print(f"  改善建议:")
        for t in result["improvement_tips"]:
            print(f"    → {t}")

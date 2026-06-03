"""
预测接口 v2.0
=============
v2 新增：基于 2026 年真实银行准入规则的产品评分层，
与 XGBoost 模型混合使用（有模型时 60% ML + 40% 规则，
无模型时 100% 规则）。
"""

import json, os
import numpy as np

BASE      = os.path.dirname(__file__)
META_PATH = os.path.join(BASE, "models/meta.json")

# ─────────────────────────────────────────────────────────────
# 真实产品规则库（来源：2026年主要扫码出额产品）
# hard_reject: 任意一条为 True → 直接拒绝（概率约 3%）
# boosts:      (条件函数, 加分权重)  正 = 加分，负 = 减分
# base:        无任何信息时的基础通过率
# ─────────────────────────────────────────────────────────────
def _mk_rules(f: dict):
    """f = form dict；返回每个产品的规则评分 dict"""
    age   = f.get("age", 30)
    inc   = max(f.get("monthly_income", 1), 1)
    pf    = f.get("provident_fund_months", 0)   # 公积金缴纳月数
    ov90  = bool(f.get("overdue_90d", 0))        # 是否有 90 天以上逾期
    ovcnt = f.get("overdue_count", 0)            # 近 2 年逾期次数
    inq6  = f.get("inquiries_6m", 0)             # 近 6 个月查询次数
    inq2m = f.get("inquiries_2y", 0)             # 近 2 年查询次数（当月近似）
    dti   = f.get("existing_monthly_payment", 0) / inc
    cc_u  = (f.get("credit_card_used", 0) / f["credit_card_limit"]
             if f.get("credit_card_limit", 0) > 0 else 0)
    prop  = f.get("has_property", False)
    sal   = f.get("is_salary_bank", False)       # 是否在该行代发工资
    exist = f.get("has_existing_product", False) # 是否有该行存量产品
    job   = f.get("job_type", "private")         # gov/state/private/self
    wy    = f.get("work_years", 0)

    # 近 2 个月查询（用 inq6 的一半近似）
    inq2mo = inq6 // 2 if inq6 > 0 else 0

    rules = {}

    # ── 1. 招商银行 闪电贷 ────────────────────────────────────
    rules["cmb_flash"] = dict(
        base=0.38,
        hard_reject=[
            age < 18 or age > 60,
            ov90,                         # 历史无 2（严重逾期）
            ovcnt >= 3,                   # 近半年无 3 次及以上逾期
            inq2mo > 3,                   # 近 2 个月不超 3 家机构
            inq6 >= 6,                    # 2 个月 < 6 次（用 6m 近似）
            dti > 0.7,
        ],
        boosts=[
            (pf >= 24,          +0.20),   # 公积金缴满 2 年
            (pf >= 12,          +0.10),
            (sal,               +0.20),   # 招行代发工资
            (exist,             +0.15),   # 有招行存量产品
            (prop,              +0.10),
            (job in ("gov","state"), +0.10),
            (inq6 <= 2,         +0.10),
            (dti < 0.3,         +0.08),
            (ovcnt == 0,        +0.10),
            (inq6 >= 4,         -0.15),
            (dti > 0.5,         -0.15),
        ],
    )

    # ── 2. 招商银行 尊享闪电贷（生意贷）──────────────────────
    rules["cmb_business"] = dict(
        base=0.30,
        hard_reject=[
            age < 18 or age > 59,
            ov90,
            ovcnt >= 3,                   # 近 3 个月不能有逾期记录
            inq6 >= 8,                    # 近 3 个月不超 4 家机构
            job != "self",                # 仅限个体/企业主
        ],
        boosts=[
            (prop,              +0.20),   # 深房加分
            (pf >= 12,          +0.10),
            (exist,             +0.15),   # 招行按揭/抵押
            (inq6 <= 2,         +0.10),
            (ovcnt == 0,        +0.12),
            (dti < 0.4,         +0.08),
        ],
    )

    # ── 3. 建设银行 建易贷（白名单单位）──────────────────────
    rules["ccb_jian"] = dict(
        base=0.35,
        hard_reject=[
            age < 20 or age > 60,
            ov90,
            ovcnt > 5,                    # 近两年逾期不超 5 个 1
            inq6 > 10,                    # 半年不超 10 次
            cc_u > 0.85,
            job not in ("gov", "state"),  # 必须是优质单位（政府/国企/500强等）
        ],
        boosts=[
            (job == "gov",       +0.25),
            (job == "state",     +0.20),
            (sal,                +0.15),
            (exist,              +0.15),
            (pf >= 24,           +0.10),
            (ovcnt == 0,         +0.12),
            (inq6 <= 4,          +0.10),
            (cc_u < 0.5,         +0.08),
            (inq6 > 6,           -0.15),
        ],
    )

    # ── 4. 建设银行 分期通 ────────────────────────────────────
    # 准入：社保基数≥4775 且本单位≥1年（用公积金+工龄近似）
    rules["ccb_install"] = dict(
        base=0.32,
        hard_reject=[
            age < 22 or age > 60,
            ov90,
            inq6 > 6,                     # 半年不超 6 次
            dti > 0.7,
            pf == 0 and wy < 1,           # 需要社保/公积金 + 工龄
        ],
        boosts=[
            (pf >= 12 and wy >= 1,  +0.20),
            (job in ("gov","state"), +0.15),
            (prop,                   +0.12),
            (sal,                    +0.10),
            (ovcnt == 0,             +0.10),
            (inq6 <= 3,              +0.10),
            (cc_u < 0.7,             +0.08),
            (dti < 0.4,              +0.08),
            (inq6 > 3,               -0.12),
            (dti > 0.5,              -0.12),
        ],
    )

    # ── 5. 民生银行 民易贷 ────────────────────────────────────
    rules["cmbc_easy"] = dict(
        base=0.34,
        hard_reject=[
            age < 22 or age > 60,
            ov90,
            ovcnt > 5,                    # 近 2 年累计逾期 ≤ 5 次
            wy < 0.5,                     # 当前单位连续工作 6 个月以上
            dti > 0.7,
        ],
        boosts=[
            (job in ("gov","state"), +0.20),
            (sal,                    +0.15),
            (pf >= 12,               +0.12),
            (exist,                  +0.10),
            (wy >= 2,                +0.10),
            (ovcnt == 0,             +0.12),
            (inq6 <= 3,              +0.08),
            (dti < 0.4,              +0.08),
            (wy < 1,                 -0.10),
            (dti > 0.5,              -0.12),
        ],
    )

    # ── 6. 中国银行 中银E贷 ───────────────────────────────────
    rules["boc_e_loan"] = dict(
        base=0.36,
        hard_reject=[
            age < 18 or age > 65,
            ov90,
            ovcnt > 6,                    # 累计逾期次数不超 6 次
            inq6 > 6,                     # 近 3 个月不超 6 次（用 6m 近似）
            pf == 0 and not sal and not exist and job not in ("gov","state"),
        ],
        boosts=[
            (pf >= 6,               +0.15),
            (pf >= 24,              +0.10),
            (sal,                   +0.18),
            (exist,                 +0.15),   # 中行按揭
            (job in ("gov","state"),+0.15),
            (ovcnt == 0,            +0.12),
            (inq6 <= 2,             +0.10),
            (dti < 0.4,             +0.08),
            (inq6 > 4,              -0.12),
            (dti > 0.5,             -0.10),
        ],
    )

    # ── 7. 中国银行 随心智贷 ─────────────────────────────────
    rules["boc_smart"] = dict(
        base=0.28,
        hard_reject=[
            age < 18 or age > 60,
            ov90,
            ovcnt > 6,
            inq6 >= 5,                    # 近 1 个月 < 5 次（用 6m 近似）
            wy < 1,                       # 本单位工作一年以上
            job == "self" and not prop,   # 一般单位需要深房
        ],
        boosts=[
            (job in ("gov","state"), +0.25),
            (sal,                    +0.20),
            (prop,                   +0.18),
            (pf >= 12,               +0.10),
            (exist,                  +0.12),
            (ovcnt == 0,             +0.12),
            (inq6 <= 2,              +0.10),
            (wy >= 3,                +0.08),
            (dti > 0.5,              -0.15),
        ],
    )

    # ── 8. 工商银行 融e借 ─────────────────────────────────────
    rules["icbc_rong_e"] = dict(
        base=0.30,
        hard_reject=[
            age < 18 or age > 60,
            ov90,
            ovcnt >= 3,
            inq2mo > 7,                   # 近 2 个月 ≤ 7 次
            dti > 0.7,
            pf == 0 and not sal and not exist and not prop,
        ],
        boosts=[
            (pf >= 24 and job in ("gov","state"), +0.30),
            (pf >= 24,              +0.20),
            (pf >= 12,              +0.10),
            (sal,                   +0.18),
            (exist,                 +0.15),   # 工行按揭/信用卡
            (prop,                  +0.10),
            (job in ("gov","state"),+0.12),
            (ovcnt == 0,            +0.12),
            (inq6 <= 2,             +0.10),
            (dti < 0.3,             +0.08),
            (inq6 >= 5,             -0.15),
            (dti > 0.5,             -0.15),
        ],
    )

    # ── 9. 农业银行 乐分易 ────────────────────────────────────
    rules["abc_easy"] = dict(
        base=0.28,
        hard_reject=[
            age < 18 or age > 60,
            ov90,
            ovcnt >= 3,                   # 半年不能有 3 个 1
            inq6 > 12,                    # 近 12 个月不超 12 次（用 6m*2 近似）
            pf == 0 and not sal and not prop and not exist,
        ],
        boosts=[
            (exist,                 +0.20),   # 农行白名单
            (pf >= 12 and inc >= 8500, +0.25),
            (pf >= 12,              +0.12),
            (sal,                   +0.15),
            (prop,                  +0.15),
            (job in ("gov","state"),+0.10),
            (ovcnt == 0,            +0.12),
            (inq6 <= 3,             +0.10),
            (dti < 0.4,             +0.08),
            (inq6 > 6,              -0.12),
            (dti > 0.5,             -0.12),
        ],
    )

    # ── 10. 车贷 ─────────────────────────────────────────────
    rules["car_loan"] = dict(
        base=0.42,
        hard_reject=[
            age < 22 or age > 60,
            ov90 and ovcnt > 4,           # 历史有 6/7 不做
            ovcnt > 6,                    # 单笔近 1 年累计逾期 ≤ 6 次
        ],
        boosts=[
            (prop,                  +0.15),
            (pf >= 12,              +0.12),
            (sal,                   +0.10),
            (job in ("gov","state"),+0.10),
            (ovcnt == 0,            +0.15),
            (inq6 <= 2,             +0.10),
            (dti < 0.4,             +0.10),
            (ov90,                  -0.30),
            (ovcnt >= 3,            -0.20),
            (inq6 > 5,              -0.10),
            (dti > 0.6,             -0.15),
        ],
    )

    # 旧名映射（向后兼容）
    rules["boc_suidaidai"] = rules["boc_e_loan"]
    rules["abc_wanlifu"]   = rules["abc_easy"]
    rules["ccb_quick"]     = rules["ccb_jian"]

    return rules


def _rule_score(product_id: str, form: dict) -> float:
    """基于规则计算通过概率 [0,1]"""
    all_rules = _mk_rules(form)
    rule = all_rules.get(product_id)
    if not rule:
        return 0.35  # 未知产品：给中等分

    # 硬拒
    if any(rule["hard_reject"]):
        return 0.03

    score = rule["base"]
    for cond, w in rule["boosts"]:
        if cond:
            score += w

    return float(np.clip(score, 0.04, 0.96))


# ─────────────────────────────────────────────────────────────
class Predictor:
    # 所有支持的产品（按大致通过率从高到低排列）
    ALL_PRODUCTS = [
        "cmb_flash", "ccb_jian", "ccb_install",
        "cmbc_easy", "boc_e_loan", "boc_smart",
        "icbc_rong_e", "abc_easy",
        "cmb_business", "car_loan",
        # 保留旧产品（前端兼容）
        "pingan_new1", "spdb_puhui", "citic_huimin",
    ]

    def __init__(self):
        import joblib
        with open(META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]
        self.job_types     = self.meta["job_types"]

        models_dir = os.path.join(BASE, "models")
        self.models = {}
        for name in self.meta["models"].keys():
            path = os.path.join(models_dir, f"{name}.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        # 旧名 → 新名映射（让旧模型也能覆盖新产品 ID）
        ALIAS = {
            "boc_e_loan":  "boc_suidaidai",
            "abc_easy":    "abc_wanlifu",
            "ccb_jian":    "ccb_quick",
        }
        for new_id, old_id in ALIAS.items():
            if new_id not in self.models and old_id in self.models:
                self.models[new_id] = self.models[old_id]

        print(f"Predictor 加载完成：{len(self.models)} 个 ML 模型")

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
        r_score = _rule_score(product_id, form)

        # 有 ML 模型：6:4 混合
        ml_model = self.models.get(product_id) or self.models.get("global")
        if ml_model:
            X     = self._extract(form)
            ml_p  = float(ml_model.predict_proba(X)[0][1])
            # 如果规则给出硬拒，ML 结果也压低
            prob  = (0.6 * ml_p + 0.4 * r_score) if r_score > 0.03 else 0.03
        else:
            prob = r_score

        prob = float(np.clip(prob, 0.03, 0.97))
        confidence = (
            "high"   if prob > 0.75 or prob < 0.25 else
            "medium" if prob > 0.6  or prob < 0.4  else
            "low"
        )
        return {
            "product":    product_id,
            "pass_prob":  round(prob, 3),
            "pass_pct":   int(prob * 100),
            "confidence": confidence,
        }

    def predict_all(self, form: dict) -> dict:
        predictions = [self.predict_one(form, pid) for pid in self.ALL_PRODUCTS]
        predictions.sort(key=lambda x: -x["pass_prob"])

        reasons = self._analyze_reasons(form)
        best    = predictions[0] if predictions else {}

        return {
            "predictions":     predictions,
            "top_product":     best.get("product"),
            "top_pass_prob":   best.get("pass_prob", 0),
            "reject_reasons":  reasons,
            "improvement_tips": self._improvement_tips(form, reasons),
        }

    def _analyze_reasons(self, form: dict) -> list[str]:
        reasons = []
        income  = form.get("monthly_income", 0)

        if form.get("overdue_90d", 0):
            reasons.append("存在90天以上严重逾期记录（绝大多数银行直接拒绝）")
        if form.get("overdue_count", 0) >= 3:
            reasons.append(f"近2年逾期{form['overdue_count']}次，超出多数银行容忍上限")
        if form.get("inquiries_6m", 0) >= 6:
            reasons.append(f"近6月征信查询{form['inquiries_6m']}次，触发银行风控")
        elif form.get("inquiries_6m", 0) >= 4:
            reasons.append(f"近6月征信查询{form['inquiries_6m']}次，偏多，建议暂停新申请")
        if income > 0:
            dti = form.get("existing_monthly_payment", 0) / income
            if dti > 0.65:
                reasons.append(f"月负债率{dti:.0%}，超过多数银行65%上限")
            elif dti > 0.5:
                reasons.append(f"月负债率{dti:.0%}，偏高，影响部分银行审批")
        if income < 4000:
            reasons.append(f"月收入{income:.0f}元，低于多数银行准入门槛")
        if form.get("credit_card_limit", 0) > 0:
            util = form.get("credit_card_used", 0) / form["credit_card_limit"]
            if util > 0.85:
                reasons.append(f"信用卡使用率{util:.0%}，过高影响评分")
        if form.get("work_years", 0) < 1:
            reasons.append("工龄不足1年，影响收入稳定性评分")
        if form.get("provident_fund_months", 0) == 0:
            reasons.append("无公积金记录，减少主流银行准入机会")
        return reasons

    def _improvement_tips(self, form: dict, reasons: list[str]) -> list[str]:
        tips = []
        for r in reasons[:3]:
            if "90天" in r:
                tips.append("严重逾期记录影响极大，建议优先尝试车贷等抵押类产品，或等待记录满5年后再申请")
            elif "逾期" in r:
                tips.append("尽快还清所有逾期欠款，保持6个月以上良好还款记录后再申请")
            elif "查询" in r:
                tips.append("停止一切新的贷款/信用卡申请，等待3-6个月让查询次数自然降低")
            elif "负债率" in r:
                tips.append("优先提前偿还小额贷款，将月负债率降至50%以下再申请")
            elif "收入" in r:
                tips.append("补充收入证明（兼职/投资等），或选择车贷、农业银行乐分易等门槛相对低的产品")
            elif "信用卡" in r:
                tips.append("将信用卡使用率降至70%以下，可临时提高授信额度或减少日常刷卡使用")
            elif "工龄" in r:
                tips.append("在当前单位再工作满1年后申请，或提供公积金/社保证明增强稳定性")
            elif "公积金" in r:
                tips.append("申请由单位缴纳公积金，缴满6个月后可大幅提升建行、工行等产品通过率")
        if not tips:
            tips.append("当前资质良好，建议优先申请通过率最高的推荐产品")
        return tips


# ── 命令行测试 ────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = Predictor()

    test_cases = [
        {
            "name": "政府单位优质客户",
            "form": {
                "age": 35, "job_type": "gov", "work_years": 8,
                "monthly_income": 12000, "provident_fund_months": 48,
                "bank_balance": 50000, "has_property": True,
                "property_value": 2000000, "property_loan": 800000,
                "existing_monthly_payment": 3000, "credit_card_used": 5000,
                "credit_card_limit": 50000, "inquiries_6m": 1,
                "inquiries_2y": 2, "overdue_count": 0, "overdue_90d": 0,
                "loan_amount": 200000, "loan_term": 36,
                "is_salary_bank": True, "has_existing_product": True,
            }
        },
        {
            "name": "私企员工中等资质",
            "form": {
                "age": 30, "job_type": "private", "work_years": 2,
                "monthly_income": 8000, "provident_fund_months": 18,
                "bank_balance": 15000, "has_property": False,
                "property_value": 0, "property_loan": 0,
                "existing_monthly_payment": 2000, "credit_card_used": 8000,
                "credit_card_limit": 20000, "inquiries_6m": 3,
                "inquiries_2y": 6, "overdue_count": 1, "overdue_90d": 0,
                "loan_amount": 100000, "loan_term": 24,
                "is_salary_bank": False, "has_existing_product": False,
            }
        },
        {
            "name": "高风险客户",
            "form": {
                "age": 28, "job_type": "private", "work_years": 0.5,
                "monthly_income": 5000, "provident_fund_months": 0,
                "bank_balance": 2000, "has_property": False,
                "property_value": 0, "property_loan": 0,
                "existing_monthly_payment": 3500, "credit_card_used": 18000,
                "credit_card_limit": 20000, "inquiries_6m": 7,
                "inquiries_2y": 12, "overdue_count": 4, "overdue_90d": 1,
                "loan_amount": 200000, "loan_term": 36,
                "is_salary_bank": False, "has_existing_product": False,
            }
        },
    ]

    for case in test_cases:
        print(f"\n{'='*55}")
        print(f"测试案例：{case['name']}")
        result = predictor.predict_all(case["form"])
        print(f"  TOP5 产品推荐:")
        for p in result["predictions"][:5]:
            bar = "█" * int(p["pass_pct"] / 5)
            print(f"    {p['product']:20s} {p['pass_pct']:3d}% {bar}")
        if result["reject_reasons"]:
            print(f"  主要风险:")
            for r in result["reject_reasons"]:
                print(f"    ⚠  {r}")
        print(f"  改善建议:")
        for t in result["improvement_tips"]:
            print(f"    → {t}")

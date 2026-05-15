"""
模型训练脚本
============
输入：channel1_community/output/synthetic_posts.jsonl（含完整 profile 字段）
输出：models/  每家银行一个 .joblib 模型 + 全局模型 + feature_names.json

运行：
  pip3 install xgboost scikit-learn joblib
  python3 train.py
  python3 train.py --eval      # 训练后打印评估报告
"""

import argparse, json, os, warnings
import numpy as np
warnings.filterwarnings("ignore")

# ── 路径 ─────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "../channel1_community/output/synthetic_posts.jsonl")
OUT_DIR = os.path.join(BASE, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 特征定义 ─────────────────────────────────────────────────
FEATURES = [
    # 用户基本面
    "age",
    "work_years",
    "monthly_income",
    "provident_fund_months",
    "bank_balance",
    # 负债
    "existing_monthly_payment",
    "credit_card_used",
    "credit_card_limit",
    # 征信
    "inquiries_6m",
    "inquiries_2y",
    "overdue_count",
    "overdue_90d",
    # 资产
    "has_property",          # bool → 0/1
    "property_value",
    "property_loan",
    # 申请参数
    "loan_amount",
    "loan_term",
    # 关系标签
    "is_salary_bank",        # bool → 0/1
    "has_existing_product",  # bool → 0/1
    # 衍生特征
    "dti",                   # debt-to-income
    "cc_utilization",        # 信用卡使用率
    "net_property_value",    # 净房产价值
    "income_per_1k_loan",    # 月收入 / 申请额（千）
]

# 职业类型 one-hot
JOB_TYPES = ["gov", "state", "private", "self", "none"]

def extract_features(profile: dict) -> list[float]:
    p = profile
    income   = max(p.get("monthly_income", 0), 1)
    cc_limit = p.get("credit_card_limit", 0)
    dti      = p.get("existing_monthly_payment", 0) / income
    cc_util  = (p.get("credit_card_used", 0) / cc_limit) if cc_limit > 0 else 0
    net_prop = p.get("property_value", 0) - p.get("property_loan", 0)
    loan_k   = max(p.get("loan_amount", 1), 1) / 1000

    base = [
        p.get("age", 30),
        p.get("work_years", 0),
        income,
        p.get("provident_fund_months", 0),
        p.get("bank_balance", 0),
        p.get("existing_monthly_payment", 0),
        p.get("credit_card_used", 0),
        cc_limit,
        p.get("inquiries_6m", 0),
        p.get("inquiries_2y", 0),
        p.get("overdue_count", 0),
        p.get("overdue_90d", 0),
        int(p.get("has_property", False)),
        p.get("property_value", 0),
        p.get("property_loan", 0),
        p.get("loan_amount", 100000),
        p.get("loan_term", 12),
        int(p.get("is_salary_bank", False)),
        int(p.get("has_existing_product", False)),
        dti,
        cc_util,
        net_prop,
        income / loan_k,
    ]
    # 职业 one-hot
    job = p.get("job_type", "private")
    base += [1 if job == jt else 0 for jt in JOB_TYPES]
    return base

ALL_FEATURES = FEATURES + [f"job_{jt}" for jt in JOB_TYPES]

# ── 加载数据 ─────────────────────────────────────────────────
def load_data(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("source") != "synthetic": continue
                if "profile" not in rec: continue
                result = rec["labels"]["result"]
                if result not in ("pass", "reject"): continue
                records.append(rec)
            except Exception:
                continue
    print(f"加载合成样本: {len(records)} 条")
    return records

def build_xy(records: list, product_filter: str = None):
    X, y, products = [], [], []
    for r in records:
        if product_filter and r.get("product") != product_filter:
            continue
        feats = extract_features(r["profile"])
        label = 1 if r["labels"]["result"] == "pass" else 0
        X.append(feats)
        y.append(label)
        products.append(r.get("product", "unknown"))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), products

# ── 训练 ─────────────────────────────────────────────────────
def train_model(X, y, name: str = "global"):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    if len(X) >= 20:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)//4),
                                    scoring="roc_auc")
        auc = cv_scores.mean()
    else:
        auc = None

    model.fit(X, y)
    return model, auc

def save_model(model, name: str):
    import joblib
    path = os.path.join(OUT_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path

# ── 主流程 ────────────────────────────────────────────────────
def main(do_eval: bool):
    records = load_data(DATA)
    if not records:
        print("没有找到合成数据，请先运行 synthetic_generator.py")
        return

    # 收集所有产品 ID
    product_ids = list(set(r.get("product","") for r in records))
    print(f"产品类型: {product_ids}\n")

    results = {}

    # 1. 全局模型（所有产品合并）
    print("训练全局模型...")
    X, y, _ = build_xy(records)
    model_global, auc = train_model(X, y, "global")
    path = save_model(model_global, "global")
    results["global"] = {"samples": len(y), "pass_rate": f"{y.mean():.1%}",
                         "cv_auc": f"{auc:.3f}" if auc else "N/A",
                         "path": path}
    print(f"  全局: {len(y)} 样本  pass_rate={y.mean():.1%}  CV-AUC={f'{auc:.3f}' if auc else 'N/A'}")

    # 2. 每个产品单独训练
    for pid in sorted(product_ids):
        if not pid: continue
        X_p, y_p, _ = build_xy(records, product_filter=pid)
        if len(X_p) < 10:
            print(f"  [{pid}] 样本不足({len(X_p)})，跳过")
            continue
        model_p, auc_p = train_model(X_p, y_p, pid)
        path_p = save_model(model_p, pid)
        results[pid] = {"samples": len(y_p), "pass_rate": f"{y_p.mean():.1%}",
                        "cv_auc": f"{auc_p:.3f}" if auc_p else "N/A",
                        "path": path_p}
        print(f"  [{pid}] {len(y_p)} 样本  pass={y_p.mean():.1%}  AUC={f'{auc_p:.3f}' if auc_p else 'N/A'}")

    # 3. 保存特征名 + 训练摘要
    meta = {
        "feature_names": ALL_FEATURES,
        "job_types":     JOB_TYPES,
        "models":        results,
    }
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 模型已保存到 {OUT_DIR}/")
    print(f"✓ meta.json: {meta_path}")

    # 4. 评估报告
    if do_eval:
        from sklearn.metrics import classification_report
        X, y, _ = build_xy(records)
        preds = model_global.predict(X)
        print("\n── 全局模型分类报告 ──")
        print(classification_report(y, preds, target_names=["reject","pass"]))

        # 特征重要性
        importances = model_global.feature_importances_
        top = sorted(zip(ALL_FEATURES, importances), key=lambda x: -x[1])[:10]
        print("TOP 10 特征重要性:")
        for feat, imp in top:
            bar = "█" * int(imp * 200)
            print(f"  {feat:30s} {imp:.4f} {bar}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="打印详细评估报告")
    args = ap.parse_args()
    main(args.eval)

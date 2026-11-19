"""Read-only CV evidence audit. No search, training or parameter changes.

Run from expanda: python3 cv_evidence_audit_v1.py
Only openpyxl (already used by the project) is required in addition to Python.
External statistics are transcribed with source/table identifiers, not raw data.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import NormalDist

N = NormalDist()
SOURCES = {
    "Chen2018": {
        "url": "https://doi.org/10.1155/2018/3747632",
        "finding": "北京城市路段的拟合比较支持对数正态；不是中欧长途货运参数标定。",
    },
    "Zhao2011": {
        "url": "https://urbanfreightlab.com/wp-content/uploads/2023/04/Truck-Travel-Time-Reliability-and-Prediction-in-a-Port-Drayage-Network.pdf",
        "finding": "作者博士论文第6章，印刷页79、Table 6.2：按距离分组的CV为0.151/0.141/0.195/0.139/0.167/0.114；正文称均值约0.15、多数行程0.1–0.3。配套期刊DOI:10.1057/mel.2011.24。",
    },
    "Lessan2018": {
        "url": "https://doi.org/10.1177/0361198118780830",
        "finding": "武汉–广州高铁区间运行时间比较倾向log-logistic；客运高铁，不能据此验证货运铁路CV=0.10。",
    },
    "Wu2020": {
        "url": "https://doi.org/10.1016/j.oceaneng.2019.106790",
        "finding": "Houston Ship Channel AIS研究：点到点穿越时间支持lognormal；不是远洋全航程CV=0.20的直接证据。",
    },
    "TTI2011": {
        "url": "https://d2dtl5nnlpfr0r.cloudfront.net/tti.tamu.edu/documents/186041-00005.pdf",
        "finding": "Vadali/Kang/Fierro, July 2011, report186041-00005, Table V-1印刷页52：BOTA口岸06–19时，21867条，49.5min/28.9min。本文以此重建两参数LN。不是原始样本P90；原报告过滤>120min记录，Fig V-10另有带位移拟合，不能宣称重现原始经验分布。",
    },
    "Carballo2016": {
        "url": "https://publications.iadb.org/publications/english/document/Endogenous-Border-Times.pdf",
        "finding": "Endogenous Border Times, Table1印刷页35/PDF页39：2013 Customs Delay均值5.2天、P50=2、P75=7、P90=13、P95=18。是Callao海运进口海关时间，非整个海港/边境总时间；同表Total Border Delay均值16.5天。",
    },
    "Li2026": {
        "url": "https://doi.org/10.1016/j.trc.2026.105699",
        "finding": "用户提供的A robust multi-objective optimisation method for airport ground movement，PDF页14/Fig7：uncertainty level=0.3时参考与增强数据的CV分布KL=0.058；不是两个运输时间对数正态的KL，更不是通用通过阈值。",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_profile(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CCP_UNCERTAINTY_PROFILES"
                for t in node.targets):
            return ast.literal_eval(node.value)["transport_border_saa_v1"]
    raise ValueError("Active uncertainty profile missing")


def capped_moment(mu, sigma, cap, order=1):
    z = (math.log(cap) - mu) / sigma
    return (math.exp(order*mu + .5*order*order*sigma*sigma)
            * N.cdf(z-order*sigma) + cap**order*N.cdf(-z))


def distribution(cv, cap=None):
    sigma = math.sqrt(math.log1p(cv*cv))
    mu = -.5*sigma*sigma
    if cap is not None:
        lo, hi = -50., math.log(cap)+50.
        for _ in range(120):
            mid = (lo+hi)/2
            if capped_moment(mid, sigma, cap) < 1:
                lo = mid
            else:
                hi = mid
        mu = (lo+hi)/2
    q90 = math.exp(mu+N.inv_cdf(.9)*sigma)
    actual_cv = cv if cap is None else math.sqrt(
        max(0., capped_moment(mu, sigma, cap, 2)-1))
    return {"mu": mu, "sigma": sigma, "input_cv": cv,
            "actual_multiplier_cv": actual_cv,
            "p90_multiplier": min(q90, cap) if cap else q90,
            "cap_factor": cap,
            "cap_mass": N.cdf((mu-math.log(cap))/sigma) if cap else 0.}


def ln_kl(p, q):
    return (math.log(q["sigma"]/p["sigma"])
            + (p["sigma"]**2+(p["mu"]-q["mu"])**2)
            / (2*q["sigma"]**2)-.5)


def table(rows):
    if not rows:
        return ""
    headers = list(rows[0])
    def cell(x):
        return html.escape(f"{x:.6g}" if isinstance(x, float) else str(x))
    return "<table><tr>"+"".join(f"<th>{cell(h)}</th>" for h in headers)+"</tr>"+"".join(
        "<tr>"+"".join(f"<td>{cell(r.get(h, ''))}</td>" for h in headers)+"</tr>"
        for r in rows)+"</table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = (args.out or repo/"transformer_ppo_runs/diagnostics/ccp_cv_evidence_audit_v1").resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output already contains files; choose another --out: {out}")
    episode = repo/"formal_icm_s0_v1/src/support/transformer_ppo/episode.py"
    engine = repo/"formal_icm_s0_v1/src/baseline_uncertainty.py"
    workbook = repo/"formal_icm_s0_v1/assets/data/data_expanded.xlsx"
    paths = [episode, engine, workbook]
    hashes = {str(p.relative_to(repo)): digest(p) for p in paths}
    profile = read_profile(episode)
    from openpyxl import load_workbook
    book = load_workbook(workbook, read_only=True, data_only=True)
    records = list(book["Node_Border"].values)
    book.close()
    borders = {}
    for mode in ("road", "rail", "water"):
        col = records[0].index(f"BorderDelay_{mode}_h")
        values = [float(r[col] or 0) for r in records[1:]]
        borders[mode] = {"positive_nodes": sum(v > 0 for v in values),
                         "nodes": len(values), "min_h": min(values), "max_h": max(values)}

    # Compare our analytic reconstruction with the actual frozen functions,
    # without importing the evaluator or invoking any transport evaluation.
    names = {"capped_lognormal_mean", "calibrated_capped_lognormal_mu"}
    nodes = [n for n in ast.parse(engine.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != 2:
        raise ValueError("Frozen cap implementation not recognized")
    env = {"math": math, "NormalDist": NormalDist}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(engine), "exec"), env)
    for cv in (.45, .5, 28.9/49.5, .7):
        cap = float(profile["border_delay_max_factor"])
        d = distribution(cv, cap)
        mu = env["calibrated_capped_lognormal_mu"](cv, cap)
        if abs(mu-d["mu"]) > 1e-10 or abs(env["capped_lognormal_mean"](mu,d["sigma"],cap)-1) > 1e-10:
            raise ValueError("Analytic calculation does not reproduce frozen code")

    rows = []
    evidence = {
        ("travel", "road"): ("Chen2018; Zhao2011", "分布有支持；CV约0.15有外部参照，非本路线实测标定"),
        ("travel", "rail"): ("Lessan2018", "LN为统一建模简化；CV是假设"),
        ("travel", "water"): ("Wu2020", "分布有局部场景支持；CV是假设"),
        ("border", "road"): ("TTI2011", "可作同均值形状参照；P90仍有偏差，待敏感性检查"),
        ("border", "rail"): ("无直接CV标定数据", "分布和CV为假设；不等于口岸均值无数据"),
        ("border", "water"): ("Carballo2016", "当前均值为0，未激活；文献海关时间不能直接代入"),
    }
    for kind in ("travel", "border"):
        for mode in ("road", "rail", "water"):
            cv = profile["mode_time_cv" if kind == "travel" else "border_delay_cv"][mode]
            cap = profile["mode_time_max_factor"][mode] if kind == "travel" else profile["border_delay_max_factor"]
            d = distribution(cv, cap)
            active = kind != "border" or borders[mode]["positive_nodes"] > 0
            source, verdict = evidence[kind, mode]
            rows.append({"来源": kind+"_"+mode, "配置CV": cv,
                         "分布": "lognormal" if cap is None else "capped mean-calibrated lognormal",
                         "上限倍数": cap if cap else "无",
                         "正均值乘子实际CV": d["actual_multiplier_cv"] if active else "不适用：均值0",
                         "P90除以均值": d["p90_multiplier"] if active else "不适用：均值0",
                         "参考": source, "判断": verdict})

    reference = distribution(28.9/49.5)
    reference_q90 = 49.5*reference["p90_multiplier"]
    road = []
    for label, cv in [("旧说明示例", .45), ("当前代码", profile["border_delay_cv"]["road"]),
                      ("文献CV候选（保留代码封顶）", 28.9/49.5), ("敏感性候选", .6)]:
        model = distribution(cv, profile["border_delay_max_factor"])
        proxy = distribution(cv)
        q90 = 49.5*model["p90_multiplier"]
        road.append({"方案": label, "输入CV": cv, "封顶后实际CV": model["actual_multiplier_cv"],
                     "参考P90_分钟": reference_q90, "代码形式P90_分钟": q90,
                     "P90有符号相对误差%": 100*(q90/reference_q90-1),
                     "P90绝对相对误差%": 100*abs(q90/reference_q90-1),
                     "未封顶辅助KL_model_to_ref": ln_kl(proxy, reference),
                     "未封顶辅助KL_ref_to_model": ln_kl(reference, proxy)})
    sigma = math.sqrt(2*math.log(5.2/2))
    sea = {"source": "Carballo2016 Table1 2013 Customs Delay", "mean_days": 5.2,
           "median_days": 2, "reference_p90_days": 13, "reference_p95_days": 18,
           "reconstructed_lognormal_cv": math.sqrt(math.exp(sigma*sigma)-1),
           "reconstructed_p90_days": 2*math.exp(sigma*N.inv_cdf(.9)),
           "reconstructed_p95_days": 2*math.exp(sigma*N.inv_cdf(.95)),
           "qualification": "文献统计量重建；不是拟合原始数据，也不是当前程序海运口岸分布"}
    for p in (90, 95):
        sea[f"p{p}_signed_error_percent"] = 100*(sea[f"reconstructed_p{p}_days"]/sea[f"reference_p{p}_days"]-1)
    sensitivity = [
        {"来源": "road travel", "CV网格": "0.10, 0.15, 0.20, 0.30", "依据": "覆盖Zhao所述多数行程0.1–0.3；不是可信区间"},
        {"来源": "rail travel", "CV网格": "0.05, 0.10, 0.15, 0.20", "依据": "基线0.10的0.5–2倍；研究者设定，非实测范围"},
        {"来源": "water travel", "CV网格": "0.10, 0.20, 0.30", "依据": "基线0.20上下50%；研究者设定，非实测范围"},
        {"来源": "road border", "CV网格": "0.30, 0.50, 0.583838, 0.70", "依据": "保留旧说明敏感性区间0.3–0.7并加入TTI外部参照"},
        {"来源": "rail border", "CV网格": "0.30, 0.50, 0.70", "依据": "基线0.50上下40%；研究者设定，非实测范围"},
        {"来源": "water border", "CV网格": "暂不启动CV扫描", "依据": "先明确时间定义及正均值；零均值乘任何因子仍为零"},
    ]
    if hashes != {str(p.relative_to(repo)): digest(p) for p in paths}:
        raise RuntimeError("Inputs changed during audit")
    out.mkdir(parents=True, exist_ok=True)
    payload = {"completed": True, "new_searches": 0, "new_objective_evaluations": 0,
               "training_updates": 0, "parameters_changed": False, "profile": profile,
               "input_sha256": hashes, "border_nodes": borders, "sources": SOURCES,
               "six_sources": rows, "road_border": road, "sea_reference": sea,
               "proposed_sensitivity_not_executed": sensitivity,
               "actual_capped_vs_uncapped_KL": "infinite in either direction: cap atom and unmatched upper support; finite KL reported above is an UNCAPPED proxy"}
    (out/"AUDIT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    for name, data in [("SIX_SOURCES.csv", rows), ("ROAD_BORDER_P90_KL.csv", road), ("SENSITIVITY_PROPOSAL.csv", sensitivity)]:
        with (out/name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    body = """<h1>六类时间不确定性：CV 与分布证据核对</h1>
<p>基于当前本地冻结引擎 transport_border_saa_v1。零新增搜索、零方案评价、零训练更新；没有修改参数或原结果。六个配置项不等于六个已激活且已实证标定的随机来源。</p>
<h2>1–2. 当前设置及证据类别</h2>"""+table(rows)+"""
<p>运输乘子为均值1的普通对数正态。口岸用 min(lognormal,3.5)，重新校准位置参数使封顶后均值为1；这是封顶而不是剔除超限样本后的条件截断。输入CV不等于封顶后的实际CV。当前三个口岸配置CV均为0.50，正均值事件的实际CV约0.49384。</p>
<p>代码按有向弧产生运输乘子、按口岸事件产生延误；同一情景中的方案共用这些抽样。文献整段行程的CV与模型单弧CV不是天然等价，线路聚合与相关性仍属适用范围限制。本次只检查边际分布和参数依据，未验证跨弧相关结构。</p>
<p>冻结工作簿有54个节点，公路/铁路各31个节点的口岸均值为正，海运全部为0。当前profile不额外加入CustomsClearance_h。节点是否用于某条路线另由边境弧映射决定；不能仅数节点推断每次搜索经过多少口岸。</p>
<h2>3. 分布检验的边界</h2>
<p>陈等的城市道路与Wu等的窄航道研究支持相应场景的分布形状，不能直接验证本研究所有线路。Lessan的客运高铁研究倾向log-logistic，使用lognormal需要说明统一建模的简化，CV=0.10仍是假设。若补分布敏感性，log-logistic与lognormal应匹配均值和CV，再比较P90及最终决策，不能把形状变化和均值变化混在一起。</p>
<p>Carballo表1已核对：5.2天均值、2天中位数、P90=13天、P95=18天属于2013 Customs Delay。按均值和中位数重建LN后，CV=2.4，P90=11.7607天（−9.53%），P95=19.4333天（+7.96%）。因此仅凭P95接近，不能断言LN已通过验证；它也不是整个港口停留时间的标定。不能把重建数据当作原始样本做拟合优劣检验。</p>
<h2>4. 公路口岸：P90为主、KL为辅</h2>"""+table(road)+"""
<p>所有方案在此表统一采用均值49.5分钟，仅比较分布形状。参考P90=85.5744分钟是由TTI表中的均值和标准差重建的两参数对数正态分位数，不是报告提供的实测P90。TTI原数据有120分钟筛选，并另展示带位移拟合；本表不是经验分布拟合检验。</p>
<p>KL方向在列名中明确，使用自然对数。有限KL仅比较未封顶的辅助对数正态；代码的封顶分布在上限有点质量，与连续未封顶参考直接比较时，严格KL两个方向均为无穷，不能套用普通LN公式。若后续要比较实际截尾规则，需对两组使用明确定义的一致规则或报告共同分箱的离散指标，并另列尾概率。</p>
<p>学长论文中的KL=0.058是其特定分布比较的结果，不是本研究可直接套用的通过阈值。P90容许偏差应由研究精度要求事先确定；目前不按事后选择阈值宣布通过。CV=0.50低于该外部参照，是否调整需结合时段、边境过程定义和敏感性结果。</p>
<h2>5. 假设参数的敏感性范围（尚未运行）</h2>"""+table(sensitivity)+"""
<p>先逐来源改变CV、保持均值与其他参数不变，沿用共同标准随机数，以固定方案检查目标P90、排序及可行性。分布设定确定后再考虑重搜索。CV证据检验与100/1000次实现的数值精度检验回答不同问题；不能用其中一个替代另一个。改参数后只另存新版本，不覆盖已完成实验。</p>
<h2>来源及适用范围</h2><ul>"""+"".join(
        f'<li><a href="{html.escape(v["url"], quote=True)}">{k}</a>：{html.escape(v["finding"])}</li>'
        for k,v in SOURCES.items())+"</ul>"
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CV evidence audit</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1280px;margin:36px auto;padding:0 20px;color:#1d2835}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border:1px solid #ccd5df;padding:8px;text-align:left}th{background:#eef3f8}h1,h2{line-height:1.3}a{color:#1565a0}</style>'+body+'</html>'
    (out/"REPORT_CN.html").write_text(document, encoding="utf-8")
    print(f"[CV-EVIDENCE] COMPLETE {out}")
    print("No new searches/evaluations/training updates; parameters unchanged.")
    print("Road border current: "+json.dumps(road[1], ensure_ascii=False))
    print("Sea customs reference: "+json.dumps(sea, ensure_ascii=False))


if __name__ == "__main__":
    main()

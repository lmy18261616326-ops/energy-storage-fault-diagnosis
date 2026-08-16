#!/usr/bin/env python3
"""Extract front matter from the local papers selected for the draft."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference" / "to work"
OUT = ROOT / "ML" / "reports" / "ieee_paper_template_2026-08-04"

NAMES = [
    "双向DC-DC开路故障-EKF多重校正诊断.pdf",
    "混合动力汽车双向DC-DC-传感器故障EKF与GLR诊断.pdf",
    "直流微电网变换器-传感器故障实时容错控制.pdf",
    "直流微电网储能变换器-开路故障诊断与拓扑重构.pdf",
    "电池储能双向DC-DC-半桥Buck-Boost双闭环控制.pdf",
    "开关变换器统一建模-状态空间平均法.pdf",
    "微电网电池接口-双向DC-DC建模与控制.pdf",
    "电池故障诊断-故障机理传感融合与人工智能综述.pdf",
    "锂电池模型故障诊断-框架与观测器综述.pdf",
    "锂电池故障诊断-模型数据融合与事件触发.pdf",
    "光储系统双向Buck-Boost-低纹波与高效率设计.pdf",
    "微电网储能接口-双向变换器拓扑与控制综述.pdf",
    "多端口双向DC-DC-电池SoC均衡与故障旁路.pdf",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {path.name: path for path in REF.rglob("*.pdf")}
    rows = []
    for name in NAMES:
        path = paths.get(name)
        if path is None:
            rows.append({"requested_name": name, "error": "not found"})
            continue
        try:
            reader = PdfReader(path)
            text = "\n".join((reader.pages[index].extract_text() or "") for index in range(min(2, len(reader.pages))))
            metadata = reader.metadata or {}
            rows.append(
                {
                    "requested_name": name,
                    "path": str(path.relative_to(ROOT)),
                    "pages": len(reader.pages),
                    "metadata": {str(key): str(value) for key, value in metadata.items()},
                    "frontmatter_text": text[:12000],
                    "last_page_text": (reader.pages[-1].extract_text() or "")[:2000],
                }
            )
        except Exception as exc:
            rows.append({"requested_name": name, "path": str(path.relative_to(ROOT)), "error": repr(exc)})
    target = OUT / "reference_frontmatter.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    for row in rows:
        print("\n===", row["requested_name"], "===")
        print(row.get("frontmatter_text", row.get("error", ""))[:1500].replace("\x00", ""))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "zipvoice_vocoder_samples_eval40"
DEFAULT_OUT_DIR = ROOT / "human_eval_manual_blind_pack"
DEFAULT_MAPPING = ROOT / "human_eval_manual_blind_pack_private_mapping.csv"
DEFAULT_ZIP = ROOT / "human_eval_manual_blind_pack.zip"


README_HEADER = """# Phiếu nghe thử A/B

Thư mục này dùng để nghe và chấm điểm các cặp audio A/B. Người nghe không cần biết file nào thuộc hệ thống nào.

## Cách nghe

1. Dùng tai nghe nếu có thể.
2. Giữ cùng một mức âm lượng cho toàn bộ bài test.
3. Với mỗi dòng `sample_XXX`, nghe:
   - `A/sample_XXX.wav`
   - `B/sample_XXX.wav`
4. Điền điểm vào bảng bên dưới hoặc file `score_sheet.csv`.

## Quy ước chấm điểm

- `MOS_A`, `MOS_B`: điểm tự nhiên của giọng nói, từ 1 đến 5.
  - 1 = rất kém
  - 2 = kém
  - 3 = trung bình
  - 4 = tốt
  - 5 = rất tốt / tự nhiên
- `Tone_pref`: bản giữ thanh điệu/phát âm tiếng Việt tốt hơn. Điền `A`, `B`, hoặc `Tie`.
- `Overall_pref`: bản nghe tổng thể tốt hơn. Điền `A`, `B`, hoặc `Tie`.
- `Notes`: ghi chú lỗi nghe được nếu có.

Người nghe có thể copy file `score_sheet.csv` thành file riêng, ví dụ `score_sheet_listener_01.csv`, rồi gửi lại.

## Bảng cần điền

| Sample | MOS_A 1-5 | MOS_B 1-5 | Tone_pref A/B/Tie | Overall_pref A/B/Tie | Notes |
|---|---:|---:|---|---|---|
"""


README_TXT_HEADER = """PHIEU NGHE THU A/B

Thu muc nay dung de nghe va cham diem cac cap audio A/B.
Nguoi nghe khong can biet file nao thuoc he thong nao.

CACH NGHE
1. Dung tai nghe neu co the.
2. Giu cung mot muc am luong cho toan bo bai test.
3. Voi moi dong sample_XXX, nghe:
   - A/sample_XXX.wav
   - B/sample_XXX.wav
4. Dien diem vao score_sheet.csv hoac score_sheet.xlsx.

QUY UOC CHAM DIEM
- MOS_A, MOS_B: diem tu nhien cua giong noi, tu 1 den 5.
  1 = rat kem, 2 = kem, 3 = trung binh, 4 = tot, 5 = rat tot / tu nhien.
- Tone_pref: ban giu thanh dieu/phat am tieng Viet tot hon. Dien A, B, hoac Tie.
- Overall_pref: ban nghe tong the tot hon. Dien A, B, hoac Tie.
- Notes: ghi chu loi nghe duoc neu co.

Nguoi nghe co the copy file score_sheet.csv hoac score_sheet.xlsx thanh file rieng,
vi du score_sheet_listener_01.csv, roi gui lai.

BANG CAN DIEN
Sample,MOS_A 1-5,MOS_B 1-5,Tone_pref A/B/Tie,Overall_pref A/B/Tie,Notes
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a blinded manual A/B listening pack.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def require_files(source_root: Path, limit: int) -> list[str]:
    baseline_dir = source_root / "baseline"
    finetuned_dir = source_root / "plus_cb"
    baseline = {p.name for p in baseline_dir.glob("*.wav")}
    finetuned = {p.name for p in finetuned_dir.glob("*.wav")}
    names = sorted(baseline & finetuned)
    if len(names) < limit:
        raise RuntimeError(
            f"Need at least {limit} matched wav files, found {len(names)} in {baseline_dir} and {finetuned_dir}"
        )
    return names[:limit]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_score_sheet(path: Path, sample_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mos_A_1_5", "mos_B_1_5", "tone_pref_A_B_Tie", "overall_pref_A_B_Tie", "notes"])
        for name in sample_names:
            writer.writerow([Path(name).stem, "", "", "", "", ""])


def write_readme(path: Path, sample_names: list[str]) -> None:
    lines = [README_HEADER.rstrip()]
    for name in sample_names:
        sample_id = Path(name).stem
        lines.append(f"| `{sample_id}` |  |  |  |  |  |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme_txt(path: Path, sample_names: list[str]) -> None:
    lines = [README_TXT_HEADER.rstrip()]
    for name in sample_names:
        sample_id = Path(name).stem
        lines.append(f"{sample_id},,,,,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_inline_cell(row: int, col: int, value: str) -> str:
    ref = f"{xlsx_col_name(col)}{row}"
    if value == "":
        return f'<c r="{ref}"/>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def write_score_sheet_xlsx(path: Path, sample_names: list[str]) -> None:
    headers = ["sample_id", "mos_A_1_5", "mos_B_1_5", "tone_pref_A_B_Tie", "overall_pref_A_B_Tie", "notes"]
    rows = [headers] + [[Path(name).stem, "", "", "", "", ""] for name in sample_names]
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(xlsx_inline_cell(row_idx, col_idx, value) for col_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    sheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <cols>
    <col min="1" max="1" width="14" customWidth="1"/>
    <col min="2" max="5" width="20" customWidth="1"/>
    <col min="6" max="6" width="45" customWidth="1"/>
  </cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
</worksheet>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="score_sheet" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles)


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "A_system",
                "B_system",
                "A_source",
                "B_source",
                "A_file_in_pack",
                "B_file_in_pack",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_zip(out_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir.parent))


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    out_dir = args.out_dir.resolve()
    mapping_path = args.mapping.resolve()
    zip_path = args.zip_path.resolve()

    sample_names = require_files(source_root, args.limit)
    reset_dir(out_dir)
    (out_dir / "A").mkdir(parents=True, exist_ok=True)
    (out_dir / "B").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    assignments = ["baseline_A"] * (len(sample_names) // 2) + ["finetuned_A"] * (len(sample_names) - len(sample_names) // 2)
    rng.shuffle(assignments)

    mapping_rows: list[dict[str, str]] = []
    for name, assignment in zip(sample_names, assignments):
        sample_id = Path(name).stem
        baseline_src = source_root / "baseline" / name
        finetuned_src = source_root / "plus_cb" / name
        a_dst = out_dir / "A" / name
        b_dst = out_dir / "B" / name

        if assignment == "baseline_A":
            shutil.copy2(baseline_src, a_dst)
            shutil.copy2(finetuned_src, b_dst)
            a_system, b_system = "baseline", "finetuned_plus_cb"
            a_source, b_source = str(baseline_src), str(finetuned_src)
        else:
            shutil.copy2(finetuned_src, a_dst)
            shutil.copy2(baseline_src, b_dst)
            a_system, b_system = "finetuned_plus_cb", "baseline"
            a_source, b_source = str(finetuned_src), str(baseline_src)

        mapping_rows.append(
            {
                "sample_id": sample_id,
                "A_system": a_system,
                "B_system": b_system,
                "A_source": a_source,
                "B_source": b_source,
                "A_file_in_pack": str(a_dst),
                "B_file_in_pack": str(b_dst),
            }
        )

    write_score_sheet(out_dir / "score_sheet.csv", sample_names)
    write_score_sheet_xlsx(out_dir / "score_sheet.xlsx", sample_names)
    write_readme(out_dir / "README.md", sample_names)
    write_readme_txt(out_dir / "README.txt", sample_names)
    write_mapping(mapping_path, mapping_rows)
    make_zip(out_dir, zip_path)

    print(f"pack_dir={out_dir}")
    print(f"zip_path={zip_path}")
    print(f"private_mapping={mapping_path}")
    print(f"n_samples={len(sample_names)}")
    print(f"A_baseline={sum(1 for row in mapping_rows if row['A_system'] == 'baseline')}")
    print(f"A_finetuned={sum(1 for row in mapping_rows if row['A_system'] == 'finetuned_plus_cb')}")


if __name__ == "__main__":
    main()

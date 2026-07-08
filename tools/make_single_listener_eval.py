from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "human_eval_single_listener"
SAMPLES = OUT / "samples"
SEED = 20260706


SYSTEMS = {
    "baseline": ROOT / "zipvoice_vocoder_samples" / "baseline",
    "plus_cb": ROOT / "zipvoice_vocoder_samples" / "plus_cb",
}


def sample_ids() -> list[str]:
    ids = []
    baseline_dir = SYSTEMS["baseline"]
    plus_cb_dir = SYSTEMS["plus_cb"]
    for path in sorted(baseline_dir.glob("sample_*.wav")):
        if (plus_cb_dir / path.name).is_file():
            ids.append(path.stem)
    if not ids:
        raise RuntimeError("No paired baseline/plus_cb samples found")
    return ids


def build_trials() -> list[dict[str, str]]:
    rng = random.Random(SEED)
    trials = []
    for idx, sample_id in enumerate(sample_ids(), start=1):
        systems = ["baseline", "plus_cb"]
        rng.shuffle(systems)
        trial_id = f"T{idx:03d}"
        a_file = f"{trial_id}_A.wav"
        b_file = f"{trial_id}_B.wav"
        shutil.copy2(SYSTEMS[systems[0]] / f"{sample_id}.wav", SAMPLES / a_file)
        shutil.copy2(SYSTEMS[systems[1]] / f"{sample_id}.wav", SAMPLES / b_file)
        trials.append(
            {
                "trial_id": trial_id,
                "sample_id": sample_id,
                "a_file": f"samples/{a_file}",
                "b_file": f"samples/{b_file}",
                "a_system": systems[0],
                "b_system": systems[1],
            }
        )
    return trials


def write_csvs(trials: list[dict[str, str]]) -> None:
    public_fields = ["trial_id", "sample_id", "a_file", "b_file"]
    key_fields = public_fields + ["a_system", "b_system"]
    rating_fields = [
        "trial_id",
        "sample_id",
        "a_file",
        "b_file",
        "mos_a",
        "mos_b",
        "tone_preference",
        "overall_preference",
        "comment",
    ]
    with (OUT / "trials_blinded.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fields)
        writer.writeheader()
        for row in trials:
            writer.writerow({field: row[field] for field in public_fields})
    with (OUT / "answer_key_do_not_open_until_finished.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=key_fields)
        writer.writeheader()
        for row in trials:
            writer.writerow({field: row[field] for field in key_fields})
    with (OUT / "ratings_template.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rating_fields)
        writer.writeheader()
        for row in trials:
            writer.writerow(
                {
                    "trial_id": row["trial_id"],
                    "sample_id": row["sample_id"],
                    "a_file": row["a_file"],
                    "b_file": row["b_file"],
                    "mos_a": "",
                    "mos_b": "",
                    "tone_preference": "",
                    "overall_preference": "",
                    "comment": "",
                }
            )


def write_html(trials: list[dict[str, str]]) -> None:
    public_trials = [
        {
            "trial_id": row["trial_id"],
            "sample_id": row["sample_id"],
            "a_file": row["a_file"],
            "b_file": row["b_file"],
        }
        for row in trials
    ]
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Single-listener blind evaluation</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 980px; margin: 32px auto; line-height: 1.45; }}
    h1 {{ font-size: 24px; }}
    .warning {{ background: #fff3cd; border: 1px solid #ffe69c; padding: 12px; border-radius: 8px; }}
    .trial {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 18px 0; }}
    .audio-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    label {{ display: block; margin: 8px 0 4px; font-weight: 600; }}
    select, textarea {{ width: 100%; padding: 6px; }}
    textarea {{ min-height: 56px; }}
    button {{ padding: 10px 16px; border: 0; background: #1f6feb; color: white; border-radius: 8px; cursor: pointer; }}
    code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Single-listener blind evaluation: baseline vs +C+B</h1>
  <div class="warning">
    <b>Lưu ý:</b> Đây là self-listening / expert pilot audit, không phải MOS đại diện nhiều người.
    Không mở <code>answer_key_do_not_open_until_finished.csv</code> trước khi chấm xong.
  </div>
  <p>Nghe A và B bằng cùng tai nghe/loa, âm lượng cố định. Chấm MOS 1–5 cho tự nhiên, chọn bản giữ thanh/phát âm tốt hơn, và chọn overall preference.</p>
  <form id="evalForm"></form>
  <button type="button" onclick="downloadCsv()">Download ratings.csv</button>
  <script>
    const trials = {json.dumps(public_trials, ensure_ascii=False, indent=4)};
    const form = document.getElementById('evalForm');
    const mosOptions = '<option value=""></option><option value="1">1 - bad</option><option value="2">2 - poor</option><option value="3">3 - fair</option><option value="4">4 - good</option><option value="5">5 - excellent</option>';
    const prefOptions = '<option value=""></option><option value="A">A</option><option value="B">B</option><option value="tie">tie</option>';
    for (const t of trials) {{
      const div = document.createElement('div');
      div.className = 'trial';
      div.innerHTML = `
        <h2>${{t.trial_id}} / ${{t.sample_id}}</h2>
        <div class="audio-row">
          <div><h3>A</h3><audio controls preload="metadata" src="${{t.a_file}}"></audio></div>
          <div><h3>B</h3><audio controls preload="metadata" src="${{t.b_file}}"></audio></div>
        </div>
        <label>MOS A</label><select name="${{t.trial_id}}_mos_a">${{mosOptions}}</select>
        <label>MOS B</label><select name="${{t.trial_id}}_mos_b">${{mosOptions}}</select>
        <label>Bản nào giữ thanh/phát âm tiếng Việt tốt hơn?</label><select name="${{t.trial_id}}_tone_preference">${{prefOptions}}</select>
        <label>Bản nào tổng thể tốt hơn?</label><select name="${{t.trial_id}}_overall_preference">${{prefOptions}}</select>
        <label>Ghi chú lỗi nghe được</label><textarea name="${{t.trial_id}}_comment"></textarea>
      `;
      form.appendChild(div);
    }}
    function csvEscape(v) {{
      v = String(v ?? '');
      if (/[",\\n]/.test(v)) return '"' + v.replaceAll('"', '""') + '"';
      return v;
    }}
    function downloadCsv() {{
      const fd = new FormData(form);
      const rows = [['trial_id','sample_id','a_file','b_file','mos_a','mos_b','tone_preference','overall_preference','comment']];
      for (const t of trials) {{
        rows.push([
          t.trial_id,
          t.sample_id,
          t.a_file,
          t.b_file,
          fd.get(`${{t.trial_id}}_mos_a`) || '',
          fd.get(`${{t.trial_id}}_mos_b`) || '',
          fd.get(`${{t.trial_id}}_tone_preference`) || '',
          fd.get(`${{t.trial_id}}_overall_preference`) || '',
          fd.get(`${{t.trial_id}}_comment`) || ''
        ]);
      }}
      const csv = rows.map(r => r.map(csvEscape).join(',')).join('\\n') + '\\n';
      const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8' }});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'ratings.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    }}
  </script>
</body>
</html>
"""
    (OUT / "rating_form.html").write_text(html, encoding="utf-8")


def write_readme(trials: list[dict[str, str]]) -> None:
    readme = f"""# Single-listener blind evaluation

Purpose: create a real but limited self-listening audit for the paper when no external listeners are available.

This is not a formal MOS test. It can be reported only as:

- single-listener pilot listening audit;
- author/expert listening sanity check;
- qualitative support for the objective metrics.

Do not report it as population MOS or human validation with multiple listeners.

## Files

- `rating_form.html`: open this in a browser and fill ratings.
- `samples/`: blinded A/B audio files.
- `trials_blinded.csv`: trial metadata without system labels.
- `answer_key_do_not_open_until_finished.csv`: maps A/B to baseline or +C+B. Do not open before rating.
- `ratings_template.csv`: manual CSV template if the HTML form is not used.
- `analyze_single_listener.py`: analyzes `ratings.csv` after rating.

## Protocol

1. Use the same headphone/monitor and fixed volume for all trials.
2. Open `rating_form.html`.
3. For every trial, listen to A and B at least once.
4. Fill MOS A, MOS B, tone/pronunciation preference, overall preference, and comments.
5. Download `ratings.csv` into this directory.
6. Run:

```bash
python analyze_single_listener.py ratings.csv
```

## Current sample count

Paired trials: {len(trials)}

This is small because only the existing ZipVoice smoke samples are available. For a stronger listening section, generate 20--40 paired utterances first.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def write_analyzer() -> None:
    code = r'''from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KEY_PATH = ROOT / "answer_key_do_not_open_until_finished.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python analyze_single_listener.py ratings.csv")
    ratings_path = Path(sys.argv[1])
    if not ratings_path.is_absolute():
        ratings_path = ROOT / ratings_path
    ratings = read_csv(ratings_path)
    key = {row["trial_id"]: row for row in read_csv(KEY_PATH)}

    mos_by_system: dict[str, list[float]] = defaultdict(list)
    tone_pref = Counter()
    overall_pref = Counter()
    comments: list[str] = []
    incomplete = []

    for row in ratings:
        trial_id = row["trial_id"]
        if trial_id not in key:
            incomplete.append(f"{trial_id}: missing answer key")
            continue
        k = key[trial_id]
        a_system = k["a_system"]
        b_system = k["b_system"]
        for side, system in [("a", a_system), ("b", b_system)]:
            value = row.get(f"mos_{side}", "").strip()
            if value:
                mos_by_system[system].append(float(value))
        for field, counter in [("tone_preference", tone_pref), ("overall_preference", overall_pref)]:
            pref = row.get(field, "").strip()
            if pref == "A":
                counter[a_system] += 1
            elif pref == "B":
                counter[b_system] += 1
            elif pref == "tie":
                counter["tie"] += 1
            else:
                counter["missing"] += 1
        comment = row.get("comment", "").strip()
        if comment:
            comments.append(f"- {trial_id}: {comment}")

    systems = sorted(set(mos_by_system) | {"baseline", "plus_cb"})
    report = {
        "n_trials": len(ratings),
        "mos": {
            system: {
                "n": len(mos_by_system.get(system, [])),
                "mean": mean(mos_by_system.get(system, [])),
                "stdev": stdev(mos_by_system.get(system, [])),
            }
            for system in systems
        },
        "tone_preference_counts": dict(tone_pref),
        "overall_preference_counts": dict(overall_pref),
        "comments": comments,
        "interpretation_limit": (
            "Single-listener audit only. Do not report as formal MOS or population-level human validation."
        ),
    }
    (ROOT / "single_listener_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Single-listener summary",
        "",
        "This is a single-listener blind audit, not a formal MOS study.",
        "",
        f"Trials: {len(ratings)}",
        "",
        "## MOS by system",
        "",
        "| System | n | Mean | SD |",
        "|---|---:|---:|---:|",
    ]
    for system in systems:
        vals = report["mos"][system]
        mean_s = "" if vals["mean"] is None else f'{vals["mean"]:.3f}'
        sd_s = "" if vals["stdev"] is None else f'{vals["stdev"]:.3f}'
        lines.append(f"| {system} | {vals['n']} | {mean_s} | {sd_s} |")
    lines += [
        "",
        "## Tone/pronunciation preference",
        "",
        "| Choice | Count |",
        "|---|---:|",
    ]
    for choice, count in sorted(tone_pref.items()):
        lines.append(f"| {choice} | {count} |")
    lines += [
        "",
        "## Overall preference",
        "",
        "| Choice | Count |",
        "|---|---:|",
    ]
    for choice, count in sorted(overall_pref.items()):
        lines.append(f"| {choice} | {count} |")
    if comments:
        lines += ["", "## Comments", "", *comments]
    lines += [
        "",
        "## How to write this in the paper",
        "",
        "Use conservative wording only: “A single-listener blinded pilot audit was used as a qualitative sanity check.”",
        "Do not call this MOS validation and do not claim statistically significant human preference.",
    ]
    (ROOT / "single_listener_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ROOT / "single_listener_summary.md")


if __name__ == "__main__":
    main()
'''
    (OUT / "analyze_single_listener.py").write_text(code, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES.glob("*.wav"):
        old.unlink()
    trials = build_trials()
    write_csvs(trials)
    write_html(trials)
    write_readme(trials)
    write_analyzer()
    print(f"Wrote {OUT}")
    print(f"Trials: {len(trials)}")


if __name__ == "__main__":
    main()

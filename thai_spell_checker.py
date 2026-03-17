"""
Thai Spell Checker จากรูปภาพและวิดีโอ
======================================
โปรแกรมตรวจจับคำผิดภาษาไทยจากไฟล์รูปภาพและวิดีโอ
แล้วสร้างรายงานเป็นไฟล์ HTML Dashboard
"""

import os
import re
import time
import html
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import easyocr
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.corpus.common import thai_words


# ─────────────────────────────────────────
# โครงสร้างข้อมูลสำหรับเก็บผลการตรวจสอบแต่ละรายการ
# ─────────────────────────────────────────
@dataclass
class CheckResult:
    file_type: str          # "image" หรือ "video"
    filename: str           # ชื่อไฟล์
    timestamp: str          # เวลา (วิดีโอ) หรือ "-" (รูปภาพ)
    raw_text: str           # ข้อความทั้งหมดที่อ่านได้จาก OCR
    wrong_words: list       # [{"word": ..., "suggestion": ...}]


# ─────────────────────────────────────────
# Class หลัก: Thai Spell Checker
# ─────────────────────────────────────────
class ThaiSpellChecker:
    """
    คลาสหลักสำหรับตรวจคำผิดภาษาไทยจากรูปภาพและวิดีโอ
    """

    # นามสกุลไฟล์ที่รองรับ
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

    def __init__(self, input_folder: str, output_html: str = "report.html",
                 video_sample_every_sec: int = 1, gpu: bool = False):
        """
        Parameters
        ----------
        input_folder         : โฟลเดอร์ที่วางไฟล์รูปภาพ/วิดีโอ
        output_html          : ชื่อไฟล์ HTML ผลลัพธ์
        video_sample_every_sec: ความถี่การสุ่มเฟรมวิดีโอ (วินาที)
        gpu                  : ใช้ GPU กับ EasyOCR หรือไม่
        """
        self.input_folder = Path(input_folder)
        self.output_html = output_html
        self.video_sample_every_sec = video_sample_every_sec

        print("กำลังโหลด EasyOCR (Thai + English)...")
        # โหลด EasyOCR รองรับภาษาไทยและอังกฤษ
        self.reader = easyocr.Reader(["th", "en"], gpu=gpu, verbose=False)

        # โหลด dictionary ภาษาไทยจาก PyThaiNLP (ราชบัณฑิตยสถาน)
        self.thai_dict = thai_words()
        print(f"โหลด Dictionary สำเร็จ ({len(self.thai_dict):,} คำ)\n")

        self.results: list[CheckResult] = []

    # ──────────────────────────────────────
    # ส่วนที่ 1: สแกนและอ่านไฟล์ใน Folder
    # ──────────────────────────────────────
    def scan_folder(self):
        """สแกนไฟล์ทั้งหมดใน input_folder แล้วส่งเข้าประมวลผล"""
        files = sorted(self.input_folder.iterdir())
        supported = [
            f for f in files
            if f.suffix.lower() in self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS
        ]

        if not supported:
            print(f"ไม่พบไฟล์รูปภาพหรือวิดีโอใน: {self.input_folder}")
            return

        print(f"พบไฟล์ทั้งหมด {len(supported)} ไฟล์\n{'─'*50}")

        for idx, filepath in enumerate(supported, 1):
            ext = filepath.suffix.lower()
            print(f"[{idx}/{len(supported)}] กำลังประมวลผล: {filepath.name}")

            if ext in self.IMAGE_EXTENSIONS:
                self._process_image(filepath)
            elif ext in self.VIDEO_EXTENSIONS:
                self._process_video(filepath)

        print(f"\n{'─'*50}")
        print(f"ประมวลผลครบแล้ว {len(self.results)} รายการ")

    # ──────────────────────────────────────
    # ส่วนที่ 2: ประมวลผลรูปภาพ
    # ──────────────────────────────────────
    def _process_image(self, filepath: Path):
        """อ่านข้อความจากรูปภาพด้วย EasyOCR แล้วตรวจคำผิด"""
        img = cv2.imread(str(filepath))
        if img is None:
            print(f"  ⚠ ไม่สามารถอ่านไฟล์: {filepath.name}")
            return

        text = self._ocr_image(img)
        if not text.strip():
            print(f"  → ไม่พบข้อความในภาพ")
            return

        wrong = self._check_spelling(text)
        self.results.append(CheckResult(
            file_type="image",
            filename=filepath.name,
            timestamp="-",
            raw_text=text,
            wrong_words=wrong
        ))
        print(f"  → ข้อความ: {text[:60]}{'...' if len(text)>60 else ''}")
        print(f"  → คำผิดที่พบ: {len(wrong)} คำ")

    # ──────────────────────────────────────
    # ส่วนที่ 3: ประมวลผลวิดีโอ (FPS Sampling)
    # ──────────────────────────────────────
    def _process_video(self, filepath: Path):
        """
        สุ่มตรวจเฟรมวิดีโอทุกๆ N วินาที (กำหนดโดย video_sample_every_sec)
        เพื่อลดภาระการประมวลผล
        """
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            print(f"  ⚠ ไม่สามารถเปิดวิดีโอ: {filepath.name}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = int(fps * self.video_sample_every_sec)
        total_samples = total_frames // sample_interval if sample_interval else 0

        print(f"  → FPS: {fps:.1f}, ทั้งหมด: {total_frames} เฟรม, "
              f"สุ่มตรวจ: {total_samples} จุด")

        seen_texts = set()  # เก็บข้อความที่ตรวจแล้วเพื่อกัน duplicate
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ตรวจเฉพาะเฟรมที่ถึงรอบสุ่ม
            if frame_idx % sample_interval == 0:
                sec = frame_idx / fps
                timestamp = str(datetime.timedelta(seconds=int(sec)))
                text = self._ocr_image(frame)

                if text.strip() and text not in seen_texts:
                    seen_texts.add(text)
                    wrong = self._check_spelling(text)
                    self.results.append(CheckResult(
                        file_type="video",
                        filename=filepath.name,
                        timestamp=timestamp,
                        raw_text=text,
                        wrong_words=wrong
                    ))
                    if wrong:
                        print(f"  → [{timestamp}] พบคำผิด {len(wrong)} คำ: "
                              f"{', '.join(w['word'] for w in wrong)}")

            frame_idx += 1

        cap.release()
        video_results = [r for r in self.results if r.filename == filepath.name]
        print(f"  → ตรวจทั้งหมด {len(video_results)} จุด")

    # ──────────────────────────────────────
    # ส่วนที่ 4: OCR ด้วย EasyOCR
    # ──────────────────────────────────────
    def _ocr_image(self, img) -> str:
        """
        อ่านข้อความจากภาพ (numpy array) ด้วย EasyOCR
        คืนค่าเป็น string ที่รวมทุกบรรทัด
        """
        results = self.reader.readtext(img, detail=0, paragraph=True)
        return " ".join(results)

    # ──────────────────────────────────────
    # ส่วนที่ 5: ตรวจคำผิดภาษาไทย
    # ──────────────────────────────────────
    def _check_spelling(self, text: str) -> list[dict]:
        """
        ตัดคำด้วย word_tokenize แล้วเช็คทีละคำว่าอยู่ใน Dictionary หรือไม่
        ถ้าไม่อยู่ → ใช้ correct() เพื่อหาคำแนะนำ
        คืนค่า list of {"word": ..., "suggestion": ...}
        """
        # กรองเฉพาะคำภาษาไทย (Unicode block ไทย: \u0E00-\u0E7F)
        thai_pattern = re.compile(r"[\u0E00-\u0E7F]+")

        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        wrong = []

        for token in tokens:
            token = token.strip()
            # ข้ามถ้าไม่ใช่คำภาษาไทย หรือสั้นเกินไป (1 ตัวอักษร)
            if not thai_pattern.fullmatch(token) or len(token) < 2:
                continue

            # ตรวจกับ Dictionary
            if token not in self.thai_dict:
                suggestion = correct(token)
                # แสดงเฉพาะกรณีที่ correct() แนะนำคำอื่น (ไม่ใช่คำเดิม)
                if suggestion and suggestion != token:
                    wrong.append({"word": token, "suggestion": suggestion})
                else:
                    # ถ้า correct() ไม่มีคำแนะนำ ก็ยังถือว่าน่าสงสัย
                    wrong.append({"word": token, "suggestion": "ไม่พบคำแนะนำ"})

        # ตัด duplicate ที่คำเดียวกัน
        seen = set()
        unique = []
        for w in wrong:
            if w["word"] not in seen:
                seen.add(w["word"])
                unique.append(w)

        return unique

    # ──────────────────────────────────────
    # ส่วนที่ 6: สร้าง HTML Dashboard
    # ──────────────────────────────────────
    def generate_html(self):
        """สร้างไฟล์ HTML รายงานผลการตรวจคำผิด"""
        total_checks = len(self.results)
        total_errors = sum(1 for r in self.results if r.wrong_words)
        total_wrong_words = sum(len(r.wrong_words) for r in self.results)
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # สร้าง rows ของตาราง
        rows_html = ""
        for r in self.results:
            # แสดง raw_text พร้อมไฮไลท์คำผิด
            highlighted = html.escape(r.raw_text)
            for w in r.wrong_words:
                escaped_word = re.escape(html.escape(w["word"]))
                highlighted = re.sub(
                    escaped_word,
                    f'<span class="wrong-word">{html.escape(w["word"])}</span>',
                    highlighted
                )

            # สร้าง badge แต่ละคำผิด
            if r.wrong_words:
                wrong_badges = " ".join(
                    f'<span class="badge-wrong">{html.escape(w["word"])}</span>'
                    for w in r.wrong_words
                )
                suggest_badges = " ".join(
                    f'<span class="badge-suggest">{'ไม่พบ' if w['suggestion'] == 'ไม่พบคำแนะนำ' else html.escape(w['suggestion'])}</span>'
                    for w in r.wrong_words
                )
                row_class = "row-error"
            else:
                wrong_badges = '<span class="badge-ok">ไม่พบคำผิด</span>'
                suggest_badges = "-"
                row_class = "row-ok"

            icon = "🖼️" if r.file_type == "image" else "🎬"
            type_badge = (
                f'<span class="type-image">IMAGE</span>'
                if r.file_type == "image"
                else f'<span class="type-video">VIDEO</span>'
            )

            rows_html += f"""
            <tr class="{row_class}">
                <td>{icon} {type_badge}</td>
                <td class="filename">{html.escape(r.filename)}</td>
                <td class="timestamp">{html.escape(r.timestamp)}</td>
                <td class="raw-text">{highlighted}</td>
                <td>{wrong_badges}</td>
                <td>{suggest_badges}</td>
            </tr>"""

        # ── HTML Template ──
        html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Thai Spell Checker — Dashboard</title>
  <style>
    /* ── Global ── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Sarabun', 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem;
    }}

    /* ── Header ── */
    .header {{
      text-align: center;
      margin-bottom: 2.5rem;
    }}
    .header h1 {{
      font-size: 2.2rem;
      font-weight: 800;
      background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -0.5px;
    }}
    .header p {{
      color: #64748b;
      margin-top: 0.4rem;
      font-size: 0.95rem;
    }}

    /* ── Summary Cards ── */
    .summary {{
      display: flex;
      gap: 1.2rem;
      justify-content: center;
      flex-wrap: wrap;
      margin-bottom: 2.5rem;
    }}
    .card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 1.4rem 2rem;
      text-align: center;
      min-width: 160px;
      transition: transform 0.2s;
    }}
    .card:hover {{ transform: translateY(-3px); }}
    .card .num {{
      font-size: 2.8rem;
      font-weight: 900;
      line-height: 1;
    }}
    .card .label {{
      font-size: 0.82rem;
      color: #94a3b8;
      margin-top: 0.4rem;
      letter-spacing: 0.5px;
    }}
    .card.blue   .num {{ color: #38bdf8; }}
    .card.red    .num {{ color: #f87171; }}
    .card.yellow .num {{ color: #fbbf24; }}
    .card.green  .num {{ color: #4ade80; }}

    /* ── Table Container ── */
    .table-wrap {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }}
    .table-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1.2rem 1.5rem;
      border-bottom: 1px solid #334155;
    }}
    .table-header h2 {{
      font-size: 1.05rem;
      font-weight: 700;
      color: #e2e8f0;
    }}
    .table-header .meta {{
      font-size: 0.8rem;
      color: #64748b;
    }}

    /* ── Table ── */
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    thead tr {{
      background: #0f172a;
    }}
    th {{
      padding: 0.9rem 1rem;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #64748b;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    td {{
      padding: 0.85rem 1rem;
      font-size: 0.88rem;
      border-bottom: 1px solid #1e2d3d;
      vertical-align: top;
    }}
    tr:last-child td {{ border-bottom: none; }}

    tr.row-error {{ background: rgba(248, 113, 113, 0.04); }}
    tr.row-error:hover {{ background: rgba(248, 113, 113, 0.09); }}
    tr.row-ok:hover {{ background: rgba(255,255,255,0.03); }}

    /* ── Cell Styles ── */
    .filename {{
      font-weight: 600;
      color: #93c5fd;
      font-size: 0.82rem;
      word-break: break-all;
      max-width: 160px;
    }}
    .timestamp {{
      font-family: monospace;
      color: #94a3b8;
      font-size: 0.82rem;
      white-space: nowrap;
    }}
    .raw-text {{
      color: #cbd5e1;
      max-width: 300px;
      line-height: 1.6;
    }}

    /* ── Wrong Word Highlight ── */
    .wrong-word {{
      color: #ff4d4d;
      font-weight: 800;
      text-decoration: underline wavy #ff4d4d;
      text-underline-offset: 3px;
    }}

    /* ── Badges ── */
    .badge-wrong, .badge-suggest, .badge-ok {{
      display: inline-block;
      padding: 0.22rem 0.6rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      margin: 2px;
    }}
    .badge-wrong {{
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .badge-suggest {{
      background: rgba(251, 191, 36, 0.12);
      color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.25);
    }}
    .badge-ok {{
      background: rgba(74, 222, 128, 0.12);
      color: #4ade80;
      border: 1px solid rgba(74, 222, 128, 0.25);
    }}

    /* ── Type Badge ── */
    .type-image, .type-video {{
      display: inline-block;
      padding: 0.18rem 0.55rem;
      border-radius: 6px;
      font-size: 0.7rem;
      font-weight: 800;
      letter-spacing: 0.5px;
    }}
    .type-image {{
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
    }}
    .type-video {{
      background: rgba(167, 139, 250, 0.15);
      color: #a78bfa;
    }}

    /* ── Footer ── */
    .footer {{
      text-align: center;
      margin-top: 2rem;
      color: #334155;
      font-size: 0.78rem;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #0f172a; }}
    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}

    /* ── Responsive ── */
    @media (max-width: 768px) {{
      body {{ padding: 1rem; }}
      .raw-text {{ max-width: 180px; }}
      table {{ font-size: 0.78rem; }}
    }}
  </style>
  <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700;800&display=swap" rel="stylesheet">
</head>
<body>

  <!-- Header -->
  <div class="header">
    <h1>🔍 Thai Spell Checker Dashboard</h1>
    <p>ตรวจสอบคำผิดภาษาไทยจากรูปภาพและวิดีโอ · สร้างเมื่อ {now}</p>
  </div>

  <!-- Summary Cards -->
  <div class="summary">
    <div class="card blue">
      <div class="num">{total_checks}</div>
      <div class="label">จุดที่ตรวจทั้งหมด</div>
    </div>
    <div class="card red">
      <div class="num">{total_errors}</div>
      <div class="label">จุดที่พบข้อผิดพลาด</div>
    </div>
    <div class="card yellow">
      <div class="num">{total_wrong_words}</div>
      <div class="label">คำที่น่าสงสัยทั้งหมด</div>
    </div>
    <div class="card green">
      <div class="num">{total_checks - total_errors}</div>
      <div class="label">จุดที่ผ่านการตรวจ</div>
    </div>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <div class="table-header">
      <h2>📋 รายงานผลการตรวจสอบ</h2>
      <span class="meta">ทั้งหมด {total_checks} รายการ</span>
    </div>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>ประเภท</th>
            <th>ชื่อไฟล์</th>
            <th>Timestamp</th>
            <th>ข้อความที่ตรวจพบ</th>
            <th>คำที่คาดว่าผิด</th>
            <th>คำที่แนะนำ</th>
          </tr>
        </thead>
        <tbody>
          {rows_html if rows_html else '<tr><td colspan="6" style="text-align:center;color:#475569;padding:3rem;">ไม่พบข้อมูลการตรวจสอบ</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    สร้างโดย Thai Spell Checker · ใช้ EasyOCR + PyThaiNLP (ราชบัณฑิตยสถาน)
  </div>

</body>
</html>"""

        output_path = Path(self.output_html)
        output_path.write_text(html_content, encoding="utf-8")
        print(f"\n✅ สร้างรายงาน HTML สำเร็จ: {output_path.resolve()}")
        return str(output_path.resolve())


# ─────────────────────────────────────────
# Entry Point — วาง Run ที่ส่วนนี้
# ─────────────────────────────────────────
def main():
    # ── ตั้งค่า: แก้ไข path ที่ต้องการ ──
    INPUT_FOLDER  = "./input_files"   # โฟลเดอร์ที่วางรูปภาพ/วิดีโอ
    OUTPUT_HTML   = "./report.html"   # ไฟล์ HTML ผลลัพธ์
    SAMPLE_EVERY  = 1                 # สุ่มตรวจวิดีโอทุก N วินาที
    USE_GPU       = False             # True ถ้ามี NVIDIA GPU

    # สร้างโฟลเดอร์ input ถ้ายังไม่มี
    Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Thai Spell Checker — OCR + PyThaiNLP")
    print("=" * 55)
    print(f"  Input  : {Path(INPUT_FOLDER).resolve()}")
    print(f"  Output : {Path(OUTPUT_HTML).resolve()}")
    print(f"  Video  : สุ่มทุก {SAMPLE_EVERY} วินาที")
    print("=" * 55 + "\n")

    checker = ThaiSpellChecker(
        input_folder=INPUT_FOLDER,
        output_html=OUTPUT_HTML,
        video_sample_every_sec=SAMPLE_EVERY,
        gpu=USE_GPU
    )

    start = time.time()
    checker.scan_folder()
    checker.generate_html()
    elapsed = time.time() - start

    print(f"\nใช้เวลาทั้งหมด: {elapsed:.1f} วินาที")

    # เปิด HTML อัตโนมัติใน Browser (Windows/Mac/Linux)
    import webbrowser
    webbrowser.open(str(Path(OUTPUT_HTML).resolve()))


if __name__ == "__main__":
    main()

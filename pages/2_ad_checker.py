"""
ตรวจสอบแอดโฆษณา — Ad Creative Checker
ส่งรูปแอดไปให้ Claude Vision ตรวจ checklist 5 ข้อ
"""

import base64, json, datetime, io, streamlit as st
import anthropic
from PIL import Image

# ─────────────────────────────────────────────
# CSS  (same dark theme as main app)
# ─────────────────────────────────────────────
CSS = """
<style>
.stApp { background: #0f172a; }

.main-title {
  font-size: 2.4rem; font-weight: 900;
  background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.2; margin-bottom: 4px;
}
.main-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 24px; }

.metric-card {
  background: #1e293b; border: 1px solid #334155;
  border-radius: 12px; padding: 16px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 900; color: #e2e8f0; }
.metric-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

.check-card {
  background: #1e293b; border: 1px solid #334155;
  border-radius: 10px; padding: 14px 18px; margin: 8px 0;
  font-size: 0.95rem; color: #cbd5e1;
}
.check-pass { border-left: 4px solid #22c55e; }
.check-fail { border-left: 4px solid #ef4444; }

.summary-box {
  background: rgba(99,102,241,0.08);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 10px; padding: 14px 18px; margin: 12px 0;
  font-size: 0.93rem; color: #c7d2fe;
}

.history-item {
  background: #1e293b; border: 1px solid #334155;
  border-radius: 8px; padding: 10px 14px; margin: 6px 0;
  font-size: 0.85rem; color: #94a3b8;
}

.overview-card {
  background: #1e293b; border: 1px solid #334155;
  border-radius: 12px; padding: 16px; text-align: center;
  margin-bottom: 8px;
}
.overview-value { font-size: 1.6rem; font-weight: 800; }
.overview-label { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
</style>
"""

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ตรวจสอบแอดโฆษณา",
    page_icon="📢",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "ad_history" not in st.session_state:
    st.session_state.ad_history = []

# ─────────────────────────────────────────────
# API Key: secrets first, then fallback to sidebar
# ─────────────────────────────────────────────
_api_key_from_secrets = ""
try:
    _api_key_from_secrets = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    pass

if _api_key_from_secrets:
    api_key = _api_key_from_secrets
    _key_source = "secrets"
else:
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = ""
    api_key = st.session_state.anthropic_api_key
    _key_source = "manual"

# ─────────────────────────────────────────────
# Rules (used in prompt & display)
# ─────────────────────────────────────────────
RULES = [
    {"id": 1, "rule": "ระยะเวลาโปรโมชั่น — ต้องมีวันที่หรือช่วงเวลาที่ชัดเจน"},
    {"id": 2, "rule": "ราคา — ต้องมีตัวเลขราคาหรือส่วนลด"},
    {"id": 3, "rule": "สิทธิ์ที่ลูกค้าได้รับ / ของแถม — ต้องระบุชัดเจน"},
    {"id": 4, "rule": "ของแถมแสดงเป็นรูปกล่องสีดำ (mystery box) ไม่ใช่ภาพสินค้าจริง"},
    {"id": 5, "rule": "มีข้อความระบุว่าของแถมไม่ซ้ำกับสินค้าที่ซื้อ"},
]

SYSTEM_PROMPT = """คุณเป็น AI ผู้เชี่ยวชาญตรวจสอบชิ้นงานโฆษณาภาษาไทย
วิเคราะห์ภาพโฆษณาที่ได้รับ แล้วตรวจตาม checklist 5 ข้อต่อไปนี้:

1. ระยะเวลาโปรโมชั่น — ต้องมีวันที่หรือช่วงเวลาที่ชัดเจน
2. ราคา — ต้องมีตัวเลขราคาหรือส่วนลด
3. สิทธิ์ที่ลูกค้าได้รับ / ของแถม — ต้องระบุชัดเจนว่าลูกค้าจะได้อะไร
4. ของแถมแสดงเป็นรูปกล่องสีดำ (mystery box) — ภาพของแถมต้องไม่แสดงสินค้าจริง ต้องเป็นกล่องสีดำ/กล่องปริศนา
5. มีข้อความระบุว่าของแถมไม่ซ้ำกับสินค้าที่ซื้อ — ให้มองหาข้อความสีเหลืองหรือสีทองขนาดเล็กที่อยู่ใกล้คำว่า 'โปรโมชั่น' หรือ 'สินค้าที่แถม' ข้อความอาจเขียนว่า 'สินค้าที่แถม ต้องไม่เป็นรายการเดียวกัน กับสินค้าที่ซื้อ' หรือคำที่มีความหมายใกล้เคียง ถ้าพบข้อความใดก็ตามที่สื่อว่าของแถมต้องไม่ซ้ำกับสินค้าที่ซื้อ ให้ถือว่าผ่าน

ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON ห้ามใส่ markdown code block:
{
  "results": [
    {"id": 1, "rule": "ระยะเวลาโปรโมชั่น", "passed": true, "detail": "อธิบายสั้นๆ"},
    {"id": 2, "rule": "ราคา", "passed": false, "detail": "อธิบายสั้นๆ"},
    ...
  ],
  "summary": "สรุปภาพรวมสั้นๆ 1-2 ประโยค",
  "score": "X/5"
}"""


# ─────────────────────────────────────────────
# Helper: resize image to fit API limit (< 4 MB base64)
# ─────────────────────────────────────────────
MAX_BASE64_BYTES = 4 * 1024 * 1024  # 4 MB


def compress_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Return (compressed_bytes, media_type) with base64 size < 4.8 MB."""
    MAX_LONG_SIDE = 2500
    TARGET_SIZE = 4.8 * 1024 * 1024  # 4.8 MB

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Step 1: resize so the longest side <= 2500px (LANCZOS for sharp text)
    w, h = img.size
    if max(w, h) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Step 2: compress JPEG starting at quality 95, reduce by 5 until < 4.8 MB
    quality = 95
    while quality >= 20:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(base64.b64encode(data)) < TARGET_SIZE:
            return data, "image/jpeg"
        quality -= 5

    # Last resort: return smallest version
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    return buf.getvalue(), "image/jpeg"


# ─────────────────────────────────────────────
# Helper: call Anthropic Claude Vision
# ─────────────────────────────────────────────
def call_vision_api(image_bytes: bytes, key: str) -> dict:
    """Send base64 image to Claude and return parsed JSON."""
    image_bytes, media_type = compress_image(image_bytes)
    b64 = base64.b64encode(image_bytes).decode()
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": "กรุณาตรวจสอบแอดโฆษณานี้ตามเกณฑ์ 5 ข้อ โดยเฉพาะข้อ 5 ให้ดูข้อความขนาดเล็กทั่วทั้งภาพให้ละเอียด ข้อความอาจเขียนว่า 'สินค้าที่แถม ต้องไม่เป็นรายการเดียวกัน กับสินค้าที่ซื้อ' หรือข้อความคล้ายกัน มักอยู่ใกล้บริเวณโปรโมชั่นหรือของแถม"},
                ],
            }
        ],
    )
    raw = message.content[0].text
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


# ─────────────────────────────────────────────
# Helper: crop top half & retry rule 5
# ─────────────────────────────────────────────
RULE5_PROMPT = """คุณเป็น AI ผู้เชี่ยวชาญตรวจสอบชิ้นงานโฆษณาภาษาไทย
ดูภาพนี้แล้วตอบคำถามเดียว:
มีข้อความที่สื่อว่า "ของแถมต้องไม่ซ้ำกับสินค้าที่ซื้อ" หรือไม่?
ให้มองหาข้อความสีเหลืองหรือสีทองขนาดเล็กที่อยู่ใกล้คำว่า 'โปรโมชั่น' หรือ 'สินค้าที่แถม'
ข้อความอาจเขียนว่า 'สินค้าที่แถม ต้องไม่เป็นรายการเดียวกัน กับสินค้าที่ซื้อ' หรือคำที่มีความหมายใกล้เคียง

ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON ห้ามใส่ markdown code block:
{"found": true, "detail": "อธิบายสั้นๆ ว่าเจอข้อความอะไร ตรงไหน"}
"""


def crop_top_half(image_bytes: bytes) -> bytes:
    """Crop the top 50% of an image and return as bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    cropped = img.crop((0, 0, w, h // 2))
    buf = io.BytesIO()
    fmt = "PNG" if img.format == "PNG" else "JPEG"
    cropped.save(buf, format=fmt, quality=95)
    return buf.getvalue()


def retry_rule5(image_bytes: bytes, key: str) -> dict | None:
    """Send cropped top-half image to check rule 5 only. Returns parsed JSON or None."""
    cropped = crop_top_half(image_bytes)
    cropped, media_type = compress_image(cropped)
    b64 = base64.b64encode(cropped).decode()
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=500,
        system=RULE5_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": "มีข้อความระบุว่าของแถมไม่ซ้ำกับสินค้าที่ซื้อหรือไม่? ดูข้อความสีเหลือง/สีทองขนาดเล็กให้ละเอียด"},
                ],
            }
        ],
    )
    raw = message.content[0].text
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────
# Helper: render single image result
# ─────────────────────────────────────────────
def render_result(data: dict):
    """Render checklist results for a single image."""
    results = data.get("results", [])
    summary = data.get("summary", "")
    score_str = data.get("score", "0/5")

    try:
        score_num = int(score_str.split("/")[0])
    except (ValueError, IndexError):
        score_num = sum(1 for r in results if r.get("passed"))

    # Score display
    c1, c2 = st.columns(2)
    with c1:
        color = "#22c55e" if score_num == 5 else "#fbbf24" if score_num >= 3 else "#ef4444"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{score_str}</div>'
            f'<div class="metric-label">คะแนนรวม</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        passed = sum(1 for r in results if r.get("passed"))
        failed = len(results) - passed
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:#22c55e">{passed} pass &nbsp; '
            f'<span style="color:#ef4444">{failed} fail</span></div>'
            f'<div class="metric-label">ผ่าน / ไม่ผ่าน</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Checklist
    for r in results:
        p = r.get("passed", False)
        icon = "✅" if p else "❌"
        css_cls = "check-pass" if p else "check-fail"
        detail = r.get("detail", "")
        st.markdown(
            f'<div class="check-card {css_cls}">'
            f"<b>{icon} ข้อ {r.get('id', '?')}: {r.get('rule', '')}</b><br>"
            f"<span style='color:#94a3b8'>{detail}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if summary:
        st.markdown(
            f'<div class="summary-box">💬 <b>สรุป:</b> {summary}</div>',
            unsafe_allow_html=True,
        )

    return score_num


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Anthropic API Key")
    if _key_source == "secrets":
        st.success("✅ ใช้ API Key จากระบบ")
    else:
        key_input = st.text_input(
            "API Key",
            value=st.session_state.anthropic_api_key,
            type="password",
            placeholder="sk-ant-...",
            label_visibility="collapsed",
        )
        if key_input != st.session_state.anthropic_api_key:
            st.session_state.anthropic_api_key = key_input
            api_key = key_input

    st.divider()
    st.markdown("### 📋 Checklist")
    for r in RULES:
        st.markdown(f"**{r['id']}.** {r['rule']}")

    if st.session_state.ad_history:
        st.divider()
        st.markdown("### 📜 ประวัติการตรวจ")
        for h in reversed(st.session_state.ad_history):
            score = h.get("score", "?/5")
            ts = h.get("timestamp", "")
            fname = h.get("filename", "")
            st.markdown(
                f'<div class="history-item">'
                f"<b>{fname}</b> — {score} &nbsp; <small>{ts}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">ตรวจสอบแอดโฆษณา</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">อัปโหลดรูปแอด แล้วให้ AI ตรวจ checklist 5 ข้อ อัตโนมัติ (รองรับหลายรูปพร้อมกัน)</p>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "อัปโหลดรูปแอดโฆษณา",
    type=["jpg", "jpeg", "png", "webp"],
    help="รองรับ .jpg .png .webp — เลือกได้หลายไฟล์",
    accept_multiple_files=True,
)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(f, caption=f.name, use_container_width=True)

col_btn, _ = st.columns([1, 3])
with col_btn:
    run = st.button("🔍 ตรวจสอบแอด", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("กรุณาใส่ Anthropic API Key ใน sidebar ก่อน")
        st.stop()
    if not uploaded_files:
        st.warning("กรุณาอัปโหลดรูปก่อน")
        st.stop()

    total = len(uploaded_files)
    all_results = []  # list of (filename, data, error)

    progress = st.progress(0, text="เริ่มตรวจสอบ...")

    for idx, uf in enumerate(uploaded_files):
        progress.progress(
            (idx) / total,
            text=f"กำลังตรวจ {uf.name} ({idx + 1}/{total})...",
        )
        try:
            raw_bytes = uf.getvalue()
            data = call_vision_api(raw_bytes, api_key)
            # Retry rule 5: if failed, crop top half and re-check
            rule5 = next((r for r in data.get("results", []) if r.get("id") == 5), None)
            if rule5 and not rule5.get("passed"):
                r5 = retry_rule5(raw_bytes, api_key)
                if r5 and r5.get("found"):
                    rule5["passed"] = True
                    rule5["detail"] = f"(ตรวจรอบ 2 จาก crop ครึ่งบน) {r5.get('detail', '')}"
                    # Recalculate score
                    passed_count = sum(1 for r in data["results"] if r.get("passed"))
                    data["score"] = f"{passed_count}/5"
            all_results.append((uf.name, data, None))
        except Exception as e:
            all_results.append((uf.name, None, str(e)))

    progress.progress(1.0, text="ตรวจเสร็จแล้ว!")

    # ── Overview summary ──
    total_checked = len(all_results)
    total_passed = 0
    total_failed = 0
    for fname, data, err in all_results:
        if err:
            total_failed += 1
        else:
            score_str = data.get("score", "0/5")
            try:
                s = int(score_str.split("/")[0])
            except (ValueError, IndexError):
                s = sum(1 for r in data.get("results", []) if r.get("passed"))
            if s == 5:
                total_passed += 1
            else:
                total_failed += 1

    st.markdown("---")
    st.markdown("### 📊 สรุปรวม")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown(
            f'<div class="overview-card">'
            f'<div class="overview-value" style="color:#e2e8f0">{total_checked}</div>'
            f'<div class="overview-label">ตรวจทั้งหมด</div></div>',
            unsafe_allow_html=True,
        )
    with oc2:
        st.markdown(
            f'<div class="overview-card">'
            f'<div class="overview-value" style="color:#22c55e">{total_passed}</div>'
            f'<div class="overview-label">ผ่านครบ 5 ข้อ</div></div>',
            unsafe_allow_html=True,
        )
    with oc3:
        st.markdown(
            f'<div class="overview-card">'
            f'<div class="overview-value" style="color:#ef4444">{total_failed}</div>'
            f'<div class="overview-label">ไม่ผ่าน / Error</div></div>',
            unsafe_allow_html=True,
        )

    # ── Per-image results in expanders ──
    for i, (fname, data, err) in enumerate(all_results):
        if err:
            with st.expander(f"❌ {fname} — Error", expanded=False):
                st.error(f"เกิดข้อผิดพลาด: {err}")
        else:
            score_str = data.get("score", "0/5")
            try:
                s = int(score_str.split("/")[0])
            except (ValueError, IndexError):
                s = 0
            icon = "✅" if s == 5 else "⚠️" if s >= 3 else "❌"
            with st.expander(f"{icon} {fname} — {score_str}", expanded=(i == 0)):
                score_num = render_result(data)
                if score_num == 5:
                    st.success("ผ่านครบ 5 ข้อ! 🎉")

            # Save history
            st.session_state.ad_history.append(
                {
                    "filename": fname,
                    "score": score_str,
                    "results": data.get("results", []),
                    "summary": data.get("summary", ""),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )

    if total_passed == total_checked and total_checked > 0:
        st.balloons()

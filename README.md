# 🔍 Thai Spell Checker

ระบบตรวจสะกดภาษาไทยจากรูปภาพและวิดีโอ สำหรับผลิตภัณฑ์เสริมอาหารบริษัท **โฮปฟูล จำกัด (Hopeful)**

ใช้เทคโนโลยี **EasyOCR + PyThaiNLP** พร้อมระบบ Auto-Correct และพจนานุกรมแบรนด์แบบครบวงจร

---

## ✨ Features

| Feature | รายละเอียด |
|---|---|
| 🖼️ OCR จากภาพ/วิดีโอ | รองรับ PNG, JPG, BMP, WebP, MP4, AVI, MOV, MKV |
| 🔤 Thai Spell Check | ตรวจสะกดด้วย PyThaiNLP + พจนานุกรม 60,000+ คำ |
| 🤖 OCR Auto-Correct | แก้ตัวอักษรที่ OCR อ่านผิดบ่อย (ผ→ม/ย, รค→คุ ฯลฯ) |
| 💬 Phrase Templates | เทียบประโยคต้นแบบ fuzzy match ≥70% → auto-correct |
| 🏷️ Brand Dictionary | พจนานุกรม 7 แบรนด์ พร้อมข้อมูลครบ (นวัตกรรม/รางวัล/ราคา) |
| 📞 Phone Validation | ตรวจว่าเบอร์โทรตรงกับสินค้าที่ปรากฏในเฟรมหรือไม่ |
| ⚖️ Unit Validation | ตรวจหน่วยนับ (เม็ด vs แคปซูล) ตามสินค้า |
| 🔧 Debug Log | แสดง correction ทุกรายการที่เกิดขึ้น ทั้งใน UI และ terminal |
| 📊 HTML Report | ดาวน์โหลด report สรุปผลเป็น HTML |

---

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

> **หมายเหตุ:** ครั้งแรกที่รัน EasyOCR จะดาวน์โหลด model ภาษาไทย (~500 MB) อัตโนมัติ

### 2. รัน Web App

```bash
cd thai_spell_checker
streamlit run app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501`

### 3. วิธีใช้งาน

1. อัปโหลดไฟล์รูปภาพหรือวิดีโอ (รองรับหลายไฟล์พร้อมกัน)
2. กด **🚀 เริ่มตรวจสอบ**
3. ดูผลลัพธ์:
   - 🔴 **คำผิดจริง** — OCR อ่านชัด แต่คำไม่อยู่ในพจนานุกรม
   - 🟠 **OCR อาจอ่านผิด** — confidence ปานกลาง ควรตรวจสอบ
   - ⚪ **OCR ไม่ชัด** — confidence ต่ำ ควรตรวจสอบต้นฉบับ
4. กด **ไม่ใช่คำผิด ✓** เพื่อเพิ่มคำเข้า Whitelist ถาวร

---

## 📁 โครงสร้างไฟล์

```
thai_spell_checker/
├── app.py                  # Streamlit web application (หลัก)
├── thai_spell_checker.py   # CLI script (ต้นฉบับ)
├── requirements.txt        # Python dependencies
│
├── brands.json             # ฐานข้อมูลแบรนด์ (7 สินค้า)
├── whitelist.json          # คำที่ไม่ต้องตรวจ (user-defined)
├── ocr_corrections.json    # ตาราง OCR อ่านผิด → ถูก
├── phones.json             # กฎการตรวจสอบเบอร์โทร
├── phrases.json            # ประโยคต้นแบบสำหรับ fuzzy match
│
└── getting_started.html    # คู่มือเริ่มต้นใช้งาน (HTML)
```

---

## 🏭 ฐานข้อมูลสินค้า Hopeful

| สินค้า | ชื่อไทย | หน่วย | เบอร์โทร |
|---|---|---|---|
| Beta Oil | เบต้าออยล์ | แคปซูล | 061-826-3693 |
| Beta Life | เบต้าไลฟ์ | แคปซูล | 063-479-1496 |
| Beta Herb | เบต้าเฮิร์บ | แคปซูล | 063-479-1496 |
| Beta X Plus | เบต้าเอ็กซ์พลัส | แคปซูล | 061-826-3693 |
| Beta Liv Pro Plus | เบต้าลีฟโปรพลัส | **เม็ด** | 063-479-1496 |
| Lab Farm | แล็บฟาร์ม | แคปซูล | 063-479-1496 |
| Beta Cal Pro Plus | เบต้าแคลโปรพลัส | แคปซูล | — |

---

## ⚙️ Sidebar Settings

### OCR Settings
| Setting | ค่า default | คำอธิบาย |
|---|---|---|
| Confidence ขั้นต่ำ | 0.5 | กรอง OCR box ที่ confidence ต่ำกว่านี้ |
| Threshold ตรวจสะกด | 0.7 | ต่ำกว่านี้ = OCR ไม่ชัด ไม่ตรวจสะกด |
| Threshold คำผิดจริง | 0.85 | สูงกว่านี้ = 🔴 คำผิดจริง |
| ความยาวคำขั้นต่ำ | 3 | คำสั้นกว่านี้จะถูกข้าม |

### Image Preprocessing
| Setting | คำอธิบาย |
|---|---|
| Upscale factor | ขยายภาพก่อน OCR (แนะนำ 2×) |
| Bilateral Denoise | กรอง noise รักษาขอบตัวอักษร |
| Adaptive Binarize | ทำขาว-ดำ เหมาะกับป้ายข้อความ |

### Phrase Templates
กำหนดประโยคต้นแบบที่ใช้บ่อยในวิดีโอ เช่น "คุมน้ำตาลให้ดี"
ถ้า OCR อ่านได้คล้าย ≥ threshold จะ auto-correct อัตโนมัติ

---

## 🔧 OCR Auto-Correct

ระบบแก้ตัวอักษรที่ OCR มักอ่านผิด:

| OCR อ่านผิด | ที่ถูกต้อง | เหตุผล |
|---|---|---|
| มู้, ยู้ | ผู้ | ผ ถูกอ่านเป็น ม/ย |
| รค | คุ | ร+ค ถูกอ่านแทน คุ |
| เเ | แ | สระแ ถูกอ่านเป็น 2 ตัว |
| น่าตาล | น้ำตาล | สระ น้ำ ถูกอ่านผิด |
| บ้, บ่ | ป้, ป่ | บ/ป สับสน |
| ดั, ดา | ตั, ตา | ด/ต สับสน |

แก้ไขเพิ่มเติมได้ที่ **Sidebar → 🔤 ตาราง OCR อ่านผิด**

---

## 📞 กฎการตรวจเบอร์โทร

```
061-826-3693  →  Beta Oil, Beta X Plus
063-479-1496  →  Beta Life, Beta Herb, Beta Liv Pro Plus, Lab Farm
```

ถ้าพบสินค้าและเบอร์โทรในเฟรมเดียวกัน ระบบจะตรวจว่าตรงกันหรือไม่
ถ้าไม่ตรงจะแสดง 📞 แจ้งเตือนสีแดง

---

## 🛠️ Development

### Requirements

```
streamlit>=1.35.0
opencv-python>=4.8.0
easyocr>=1.7.1
pythainlp>=5.0.0
pandas>=2.0.0
numpy>=1.24.0
```

### รัน CLI version (ไม่ใช้ Streamlit)

```bash
python thai_spell_checker.py --input image.jpg
```

---

## 📄 License

Private — บริษัท โฮปฟูล จำกัด (Hopeful Co., Ltd.)

# align_engine

ترازبندیِ خودکارِ کلمه‌به‌کلمه‌ی اسکریپت‌های صوتیِ کتاب‌ها (مرحله‌ی اولِ صوت:
align_engine ← AudioMarkerReview ← CustomLayoutGenerator).

- **نصب (یک‌بار):** دوبار کلیک روی `setup.bat`
- **اجرا:** دوبار کلیک روی `run.bat`، یا F5 در VS Code. سندِ اسکریپت‌ها و پوشه‌ی صوت را
  می‌پرسد، بعد شماره‌ی ترک‌ها را (`26`، `26-31`، `new`، `all`).
- **خروجی:** `<پوشه‌ی صوت>\output\<نامِ فایل>_draft.json`. AudioMarkerReview آن را خودش بار
  می‌کند.

**راهنمای کامل:** [GUIDE.md](GUIDE.md)

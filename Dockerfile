# 1. Gunakan base image Python yang ringan
FROM python:3.11-slim

# 2. Set environment variables untuk Python di Docker
# Mengoptimalkan TensorFlow agar membatasi jumlah thread (mencegah crash OOM di RAM 500MB Railway)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

# 3. Tetapkan direktori kerja utama di dalam container
WORKDIR /app

# 4. Buat user non-root (Hugging Face Spaces mewajibkan ini untuk keamanan)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# 5. Salin requirements.txt terlebih dahulu untuk efisiensi caching
COPY --chown=user requirements.txt /app/

# 6. Install dependensi Python (tanpa cache agar image tetap ramping)
RUN pip install --no-cache-dir --user -U pip && \
    pip install --no-cache-dir --user -r requirements.txt

# 7. Salin seluruh isi folder project ke dalam container
COPY --chown=user . /app/

# 8. Buka port default (Railway menggunakan dynamic port, HF menggunakan 7860)
EXPOSE 7860

# 9. Jalankan Flask app menggunakan Gunicorn pada port dinamis $PORT (default 7860 jika tidak diset)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} app:app"]

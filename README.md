# Machine-Learning-to-detect-bottle-caps
Project ini merupakan implementasi end-to-end Machine Learning Pipeline untuk mendeteksi warna tutup botol (light blue, dark blue, others) menggunakan YOLOv8, Python CLI, CI/CD, Docker, dan konfigurasi YAML.

#Fitur Utama

1. Model Training & Inference melalui CLI:
bsort train --config settings.yaml
bsort infer --config settings.yaml --image sample.jpg
2. Tracking eksperimen menggunakan Weights & Biases (wandb)
3. CI/CD GitHub Actions (linting, formatting, unit tests, Docker build)
4. Konfigurasi fleksibel melalui settings.yaml
4. Struktur modular untuk memudahkan pengembangan

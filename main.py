import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
from skimage import io, img_as_ubyte

# Veri Seti Dizini ve Çıktı Dizini
input_dir = "DB"  # nii.gz dosyalarının olduğu dizin
output_image_dir = "dataset/image"
output_label_dir = "dataset/label"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# İlgili Etiketi Belirleme (6 olarak örnek alınmıştır)
target_label = 10  # Projenizde hedef organın etiket değeri

# Mevcut Dosyaları Listeleme
all_files = os.listdir(input_dir)
image_files = sorted([f for f in all_files if f.startswith("amos_0") and f.endswith(".nii.gz")])
label_files = sorted([f for f in all_files if f.startswith("amos_img") and f.endswith(".nii.gz")])

# İkili Eşleştirme ve İşleme
for img_file, label_file in zip(image_files, label_files):
    # Dosya yolları
    img_path = os.path.join(input_dir, img_file)
    label_path = os.path.join(input_dir, label_file)

    # Veri Yükleme
    img = nib.load(img_path).get_fdata()  # 3D Görüntü
    label = nib.load(label_path).get_fdata()  # 3D Etiket

    # Coronal Görünüm için Transpoz İşlemi
    coronal_img = np.transpose(img, (2, 0, 1))  # Ekseni değiştir
    coronal_label = np.transpose(label, (2, 0, 1))  # Aynı işlem etiket için

    # Orta Coronal Slice Seçimi
    mid_slice = coronal_img.shape[2] // 2  # Orta slice'ı seç
    img_coronal_mid = coronal_img[:, :, mid_slice]  # Görüntü
    label_coronal_mid = coronal_label[:, :, mid_slice]  # Etiket

    # İlgili Etiketi Maskeleme
    label_coronal_mid_masked = (label_coronal_mid == target_label).astype(np.uint8) * 255

    # Görüntüleri 768x768 Boyutuna Yeniden Boyutlandırma
    img_coronal_resized = resize(img_coronal_mid, (768, 768), anti_aliasing=True)
    label_coronal_resized = resize(label_coronal_mid_masked, (768, 768), anti_aliasing=False, order=0)

    # Normalize Ederek uint8 Formatına Dönüştürme
    img_coronal_normalized = (img_coronal_resized - img_coronal_resized.min()) / (img_coronal_resized.max() - img_coronal_resized.min())
    img_uint8 = img_as_ubyte(img_coronal_normalized)
    label_uint8 = label_coronal_resized.astype(np.uint8)

    # Çıktı Dosya Adlarını Belirleme
    img_name = os.path.splitext(img_file)[0] + ".png"
    label_name = os.path.splitext(label_file)[0] + ".png"

    # PNG Olarak Kaydetme
    io.imsave(os.path.join(output_image_dir, img_name), img_uint8)
    io.imsave(os.path.join(output_label_dir, label_name), label_uint8)

print("Tüm dilimler başarıyla işlendi ve kaydedildi.")

import argparse
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips


class Metrics():
    def __init__(self) -> None:
        self.lpips_model = lpips.LPIPS(net='alex')

    def calculate_psnr_ssim(self, gt_image, dt_image):
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        dt_image = cv2.cvtColor(dt_image, cv2.COLOR_BGR2RGB)

        psnr_value = psnr(gt_image, dt_image, data_range=gt_image.max() - gt_image.min())
        ssim_value = ssim(gt_image, dt_image, win_size=5, channel_axis=2, data_range=gt_image.max() - gt_image.min())

        return psnr_value, ssim_value

    def calculate_lpips(self, gt_image, dt_image):
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        dt_image = cv2.cvtColor(dt_image, cv2.COLOR_BGR2RGB)

        gt_tensor = lpips.im2tensor(gt_image)  # Конвертация изображения в тензор
        dt_tensor = lpips.im2tensor(dt_image)

        lpips_value = self.lpips_model(gt_tensor, dt_tensor)
        return lpips_value.item()

    def calculate_metrics(self, gt_images_path, dt_images_path):
        gt_images = sorted(os.listdir(gt_images_path))
        dt_images = sorted(os.listdir(dt_images_path))

        psnr_list = []
        ssim_list = []
        lpips_list = []

        for gt_image_name, dt_image_name in zip(gt_images, dt_images):
            gt_image_path = os.path.join(gt_images_path, gt_image_name)
            dt_image_path = os.path.join(dt_images_path, dt_image_name)

            gt_image = cv2.imread(gt_image_path, cv2.IMREAD_COLOR)
            dt_image = cv2.imread(dt_image_path, cv2.IMREAD_COLOR)

            # Рассчет PSNR и SSIM
            psnr_value, ssim_value = self.calculate_psnr_ssim(gt_image, dt_image)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)

            # Рассчет LPIPS
            lpips_value = self.calculate_lpips(gt_image, dt_image)
            lpips_list.append(lpips_value)

            print(f"Image: {gt_image_name} -> PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)

        print(f"\nAverage PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt_images', type=str, required=True, help="Path to the folder with ground truth images.")
    parser.add_argument('-dt_images', type=str, required=True, help="Path to the folder with distorted/generated images.")

    args = parser.parse_args()

    metrics  = Metrics()
    metrics.calculate_metrics(args.gt_images, args.dt_images)


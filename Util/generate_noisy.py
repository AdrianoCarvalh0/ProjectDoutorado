import cv2
import matplotlib.pyplot as plt
import functions
from pathlib import Path
import sys

#windows
sys.path.insert(0, r"C:\Users\adria\Documents\Doutorado\ProjetoDoutorado")
root_dir = Path(r"C:\Users\adria\Documents\Doutorado\ProjetoDoutorado")

#linux
#sys.path.insert(0, "/home/adriano/projeto_mestrado/modules/")
#root_dir = f"/home/adriano/projeto_mestrado/modules"

root_img = '/images/'
img = 'Lenna.png'

# Loading the example image (replace 'example.jpg' with the path to your image)
original_image = cv2.imread(f'{root_dir}/{root_img}/{img}', cv2.IMREAD_GRAYSCALE)

# Adding salt and pepper noise to the image
noisy_image_salt = functions.add_salt_and_pepper_noise(original_image)
cv2.imwrite('ruidosa_salt_pepper.png',noisy_image_salt)


# Adding gaussian noise to the image
noisy_image_gauss = functions.add_gaussian_noise(original_image)
cv2.imwrite('ruidosa_gauss.png',noisy_image_gauss)

# Displaying the original and noisy images
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), 'gray', vmin=0, vmax=60)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(noisy_image_salt, cv2.COLOR_BGR2RGB),'gray', vmin=0, vmax=60)
plt.title('Image with Salt and Pepper Noise')
plt.axis('off')
plt.show()


# plt.figure(figsize=(20, 10))

# plt.imshow(original_image,'gray', vmin=0, vmax=155)
# plt.axis('off')
# plt.plot()

# plt.savefig('original.png', format='png')

# plt.figure(figsize=(20, 10))
# plt.imshow(noisy_image_gauss,'gray', vmin=0, vmax=155)
   
# plt.axis('off')
# plt.plot()
# plt.savefig('noisy_image_gauss.png', format='png')
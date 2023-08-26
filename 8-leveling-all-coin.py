import cv2
import matplotlib.pyplot as plt
def level_coins(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
def calculate_area(contour):
    return cv2.contourArea(contour)
def main():
    image = cv2.imread('coin2.jpg')

    if image is None:
        print("Image not found.")
        return
    
    leveled_image = level_coins(image)
    contours, _ =cv2.findContours(leveled_image, cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)
    
    area=0
    for i, contour in enumerate(contours):
        area =area+ calculate_area(contour)
    print(f"Total Coins area: {area}")

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Image with Contours')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

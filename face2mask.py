import cv2
import numpy as np
import torch
from PIL import Image


def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def detect_faces(image):
    # Read the image
    # image = cv2.imread('people1.jpg')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    # Create a black and white mask image
    mask = np.zeros_like(gray)

    # Loop over all detected faces
    for x, y, w, h in faces:
        # Draw rectangles around the detected faces
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Set the corresponding region in the mask image to white
        mask[y : y + h, x : x + w] = 255

    # Display the number of faces detected
    print("Faces Detected:", len(faces))

    mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA))

    return mask


class FaceToMaskCopy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("MASK",)
    # RETURN_NAMES = ("WIDTH","HEIGHT","X","Y",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab_copy/Mask"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    def run(self, image):
        # print(image)
        im = tensor2pil(image)
        mask = detect_faces(im)

        mask = pil2tensor(mask)
        channels = ["red", "green", "blue", "alpha"]
        mask = mask[:, :, :, channels.index("green")]

        return (mask,)

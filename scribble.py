from PIL import Image
import numpy as np


a = "./aachen_000000_000019_gtFine_labelIds.png"
b = Image.open(a).resize((512, 256))

b = np.array(b)

for i in range(256):
    for j in range(512):
        if b[i, j] == 20:
            b[i, j] = 7
        elif b[i, j] == 21:
            b[i, j] = 8

Image.fromarray(b).save("./aachen_000000_000019_gtFine_labelIds_.png")

# b.save("./aachen_000000_000019_gtFine_labelIds_.png")
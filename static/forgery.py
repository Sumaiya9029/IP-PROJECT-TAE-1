import cv2
import numpy as np

# Read image
img = cv2.imread("forged.jpg")

if img is None:
    print("Image not found!")
    exit()

# Resize image
img = cv2.resize(img, (600, 400))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Parameters
block_size = 16
step = 8
threshold = 5.0

blocks = []
positions = []

h, w = gray.shape

# Extract overlapping blocks
for y in range(0, h - block_size + 1, step):
    for x in range(0, w - block_size + 1, step):

        block = gray[y:y + block_size,
                     x:x + block_size]

        # Ignore flat blocks
        if np.std(block) < 10:
            continue

        mean = np.mean(block)
        std = np.std(block)

        feature = (block - mean) / std

        blocks.append(feature.flatten())
        positions.append((x, y))

blocks = np.array(blocks)

# Sort blocks
order = np.lexsort(blocks.T)
blocks = blocks[order]
positions = np.array(positions)[order]

mask = np.zeros((h, w), dtype=np.uint8)

matches = 0

# Compare neighbouring blocks
for i in range(len(blocks) - 1):

    distance = np.linalg.norm(
        blocks[i] - blocks[i + 1]
    )

    if distance < threshold:

        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]

        # Calculate distance between blocks
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # Ignore nearby blocks
        if dx < block_size * 3 and dy < block_size * 3:
            continue

        # Mark matched blocks
        cv2.rectangle(
            mask,
            (x1, y1),
            (x1 + block_size, y1 + block_size),
            255,
            -1
        )

        cv2.rectangle(
            mask,
            (x2, y2),
            (x2 + block_size, y2 + block_size),
            255,
            -1
        )

        matches += 1

# Remove small noise
kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# Remove small connected components
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mask,
    8
)

clean_mask = np.zeros_like(mask)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area >= 150:
        clean_mask[labels == i] = 255

mask = clean_mask

# Create result
result = img.copy()

# Red highlight
result[mask > 0] = [0, 0, 255]

# Blend result with original
output = cv2.addWeighted(
    img,
    0.70,
    result,
    0.30,
    0
)

# Display
cv2.imshow("Original Image", img)
cv2.imshow("Detected Mask", mask)
cv2.imshow("Forgery Detection", output)

print("Matching blocks detected:", matches)

if matches > 0:
    print("Possible Copy-Move Forgery Detected!")
else:
    print("No significant Copy-Move Forgery Detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
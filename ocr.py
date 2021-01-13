from __future__ import print_function
import cv2
import numpy as np

# Step 1: Manually define region of interest for invoice

invoice_subject  = ['your detail', 'client detail', 'invoice number & invoice date', 'product', 'invoice summary']
subject_top_left = [(28,93),(290,93),(29,219),(26,324),(314,479)]
subject_bottom_right = [(280,200),(540,206),(223,273),(538,397),(538,590)]

# Step 2: align reference picture and input picture

refFilename = "invoice.png"
imFilename = "invoice_raw1.jpg"
concate_img = "matches.jpg"
outFilename = "aligned.jpg"
output_text_file = "ocr_output.txt"

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite(concate_img, imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h


if __name__ == '__main__':

  # Read reference image
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg, h = alignImages(im, imReference)

  # Write aligned image to disk.

  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)

# Step 3: Use pytesseract as ocr_engine to detect character on the invoice
try:
    from PIL import Image
except:
    import Image
import pytesseract


def ocr_core(filename,left,top,right,bottom):
    cropped = Image.open(filename).crop((left,top,right,bottom))
    text = pytesseract.image_to_string(cropped)
    cropped.show()    #Show cropped roi
    return text

# Step 4: Write result into a text file
f = open(output_text_file, "w")
f = open(output_text_file, "a")

for i,subject in enumerate(invoice_subject):
    # print("\n" + subject+":\n")
    # print(ocr_core(outFilename,subject_top_left[i][0],subject_top_left[i][1],subject_bottom_right[i][0],subject_bottom_right[i][1]))
    f.write("\n" + subject+":\n")
    f.write(ocr_core(outFilename,subject_top_left[i][0],subject_top_left[i][1],subject_bottom_right[i][0],subject_bottom_right[i][1]))
    f.write("---------------------------------------------------------")
f.close()
print(f"ocr done. Result output in {output_text_file}")

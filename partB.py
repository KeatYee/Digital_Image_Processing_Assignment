import numpy as np
import cv2
from matplotlib import pyplot as plt


def process_img(image):
    #Read image
    img = cv2.imread(image, 0)
    
    #Switch to binary image
    _, bi_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    
    #plot put binary image
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(bi_img, cmap='gray')
    plt.title("page")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    return bi_img


    
def find_gaps(hist, min_width=60,multiplier=0.3):
    treshold = np.max(hist) * multiplier
    
    gaps = []
    in_gap = False
    
    #Iterate over x in the vertical histogram
    for x, val in enumerate(hist):
        #if value lesser than treshold, start of a potential gap
        if val < treshold and not in_gap:
            start_g = x 
            in_gap = True
        #if value more than treshold, end of a potential gap
        elif val >= treshold and in_gap:
            end_g = x
            
            #get the width of the potential gap
            gap_width = end_g - start_g

            #Only add gap that are large enough
            if gap_width >= min_width:
                gaps.append((start_g, end_g))
            in_gap = False
            
    #Handle the case where a gap continues to the far right edge of the image
    if in_gap:
        end_g = len(hist)
        gap_width = end_g - start_g
        if gap_width >= min_width:
            gaps.append((start_g, end_g))
    
    return gaps
    

def get_region(gaps,hist, min_length = 50):
    regions = []
    r_start = 0
    padding = 20
    limit = len(hist)
    
    #get the actual region of text using the gaps
    if gaps:
        #iterate over each gap
        for x, (start_g, end_g) in enumerate(gaps):
            #prevent overlap
            if start_g > r_start:
                r_length = start_g - r_start
                if (r_length >= min_length):
                    #add in padding
                    start = max(r_start - padding, 0)
                    end = min(start_g + padding, limit)
                   
                    #r_start indicate the start of next region
                    #start_gap indicate the end of region, start of new gap
                    regions.append((start, end))
    
            #set starting point for next region
            r_start = end_g + 1
    else: 
        start = max(r_start - padding, 0)
        end = min(len(hist) + padding, limit)
        regions.append((start, end)) 
     
    return regions

   
#Find columns with text 
def extract_column(image,m=0.15):
    #switch to binary image (process_img)
    binary_img = process_img(image) 
    
    #vertical histogram (black pixel)
    verti_hist = np.sum(binary_img == 0, axis=0)
    
    #find gaps
    gaps = find_gaps(verti_hist,60,m)
    print(f"Gaps between column: {gaps}")
    
    #find column        
    cols = get_region(gaps, verti_hist)
    print(f"Columns: {cols}")
    
    
    #Turn to binary image
    column_imgs = [binary_img[:, start:end] for start, end in cols]
    
    #Plot out the histogram
    plt.figure(figsize=(12, 4))
    plt.plot(verti_hist)
    plt.title("Vertical Histogram Projection (Columns)")
    plt.xlabel("Column")
    plt.ylabel("Sum of Black Pixels")
    plt.grid(True)
    plt.show()

    return column_imgs

#identify diagram
def is_diagram(img,min_r=0.05, max_r=0.12): 
    height, width = img.shape
    #get amount of black pixel
    black_pixel_count = np.sum(img == 0)
    total_pixels = height * width
    
    if total_pixels == 0:
        return False
    
    #get the ration of black pixel : total pixel
    black_ratio = black_pixel_count / total_pixels

    if black_ratio < min_r or black_ratio > 0.12:
        return True
    return False

def extract_paragraph(image,m=0.15):
    all_para_imgs = []
    col_imgs = extract_column(image,m)
    
    #Iterate through each column images extracted
    for i, col_img in enumerate(col_imgs):
        #list of paragraph within the column (requirement ii)
        para_imgs_col = []
        
        #histogram
        hori_hist = np.sum(col_img == 0, axis=1)
        gaps = find_gaps(hori_hist,30,m)
        print(f"Gaps between paragraph: {gaps}")
        
        #Find paragraph
        paras = get_region(gaps, hori_hist)
        print(f"Paragraphs: {paras}")
     
        #Turn to binary image
        para_imgs = [col_img[start:end, :] for start, end in paras]
        
        for para_img in para_imgs:
            if not is_diagram(para_img):
                #append the paragraph within the column (requirement ii)
                para_imgs_col.append(para_img)
                #append the list
        all_para_imgs.append(para_imgs_col)
    
      #Plot out the histogram
        plt.figure(figsize=(12, 4))
        plt.plot(hori_hist)
        plt.title("Horizontal Histogram Projection (Rows)")
        plt.xlabel("Row")
        plt.ylabel("Sum of Black Pixels")
        plt.grid(True)
        plt.show()
        
    #show out all paragraph

    for i, para_imgs in enumerate(all_para_imgs):
        para_nums = len(para_imgs)
        plt.figure()
        for p_i, para_img in enumerate(para_imgs):
            #plot out the graph
            plt.subplot(para_nums, 1, p_i + 1)
            plt.imshow(para_img, cmap='gray')
            plt.title(f"Column {i+1} - Paragraph {p_i+1}")
            plt.axis('off')
            
            #save the paragraph
            cv2.imwrite(f"Column {i+1}_Paragraph {p_i+1}.png", para_img)
            cv2.waitKey()

         
        plt.tight_layout()
        plt.show()
    cv2.destroyAllWindows()
    
    return all_para_imgs

    
        
  
def main():
    while True:
        print("Choose a page to pick from:\n[1] 001.png\n[2] 002,png\n[3] 003.png\n[4] 004.png\n[5] 005.png\n[6] 006,png\n[7] 007.png\n[8] 008.png")
        try:
            choice= int(input("Please enter an integer:"))
            
            match choice:
                case 1:
                   extract_paragraph("001.png")
                   break
                case 2:
                    extract_paragraph("002.png")
                    break
                case 3:
                    extract_paragraph("003.png")
                    break
                case 4:
                    extract_paragraph("004.png",0.3)
                    break
                case 5:
                   extract_paragraph("005.png")
                   break
                case 6:
                    extract_paragraph("006.png")
                    break
                case 7:
                    extract_paragraph("007.png")
                    break
                case 8:
                    extract_paragraph("008.png",0.3)
                    break
                case _:
                    print("Choose an integer between 1-8")
            
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    
if __name__ == "__main__":
    main()
          

    
    
    
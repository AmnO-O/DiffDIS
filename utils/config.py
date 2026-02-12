import os

diste1 = '/home/exp/Documents/1st_star/YQ/DIS-main/DIS5K/DIS-TE1/'
diste2 = '/home/exp/Documents/1st_star/YQ/DIS-main/DIS5K/DIS-TE2/'
diste3 = '/home/exp/Documents/1st_star/YQ/DIS-main/DIS5K/DIS-TE3/'
diste4 = '/home/exp/Documents/1st_star/YQ/DIS-main/DIS5K/DIS-TE4/'
disvd = '/home/exp/Documents/1st_star/YQ/DIS-main/DIS5K/DIS-VD/'


# Đường dẫn đến thư mục chứa ảnh bạn muốn tách nền (Inference)
# Bạn nên tạo một thư mục ngoài Desktop cho dễ tìm, ví dụ: 'test_images'
dataset_path = r'C:\Users\LAPTOP_CUA_NAM\Dischotomous_Image_Segmentation\test_images' 

# Các dòng dưới này là dành cho bộ dữ liệu DIS5K (nếu bạn không train thì không cần quá quan trọng)
# Nhưng để code không lỗi, hãy trỏ chúng về thư mục dataset_path ở trên hoặc để trống
diste1 = dataset_path
diste2 = dataset_path
diste3 = dataset_path
diste4 = dataset_path
disvd = dataset_path


diste1 = os.path.join(diste1)
diste2 = os.path.join(diste2)
diste3 = os.path.join(diste3)
diste4 = os.path.join(diste4)
disvd = os.path.join(disvd)

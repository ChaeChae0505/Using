### Computer Vision
- OpenCV, Keras, PIL, Numpy , scikit-Image, matplotlib를 위주로 
- [ ] Binalization (thresh)
- [ ] point clicking for making dataset 

# Vision task
## Library of python vision 
1. PIL(Pillow, Python Image) : slow,,,
2. Sckit-Image : Scipy , Numpy based
3. Opencv : 컴퓨터 비전 기능의 대부분을 포함하고 있음 
 - imread -> numpy array 로 변환해 저장 -> RGB 가 아닌 BGR로 로드됨
4. Open3D : PCL library 보다 좀 더 접근성이 높은 3D visualization and point cloud 처리 기능이 있는 library


## Image load 부분 차이
### PIL을 이용한 이미지 로드 
```python
import PIL import Image
import matplotlib.pyplot as plt

# Image open  image file을 ImageFile 객체로 생성함
DIR = 'test.jpg'
img = Image.open(DIR)
img_gray = Image.Open(DIR).convert('L')

# image load
fig = plt.figure()
rows = 1
cols = 2

axis_1 = fig.add_subplot(rows, cols, 1)
axis_1.imshow(img)
axis_1.set_title('PIL img')
axis_1.axis("off")
 
axis_2 = fig.add_subplot(rows, cols, 2)
axis_2.imshow(img_gray, cmat='gray')
axis_2.set_title('PIL gray')
axis_2.axis("off")
 
plt.show()
# image type : <class 'PIL.JpegImagePlugin.JpegImageFile'>
```

### skimage로 이용한 이미지 로드
```python
import skimage import io
import matplotlib.pyplot as plt

# Image open  image file을 ImageFile 객체로 생성함
DIR = 'test.jpg'
img = io.imread(DIR)
img_gray = Image.Open(DIR).convert('L')

# image load
fig = plt.figure()
rows = 1
cols = 2

axis_1 = fig.add_subplot(rows, cols, 1)
axis_1.imshow(img)
axis_1.set_title('PIL img')
axis_1.axis("off")
 
axis_2 = fig.add_subplot(rows, cols, 2)
axis_2.imshow(img_gray, cmat='gray')
axis_2.set_title('PIL gray')
axis_2.axis("off")
 
plt.show()
# image type : <class 'PIL.JpegImagePlugin.JpegImageFile'>
```





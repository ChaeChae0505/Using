"""
manually get the points!
and save the csv!
"""
import os, cv2
import tkinter
import numpy as np

clicks = list()
click = list()
import csv


# %%Functions
## 'PLease dont modify this section'
def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global click
        click.append(x)
        click.append(y)
        print(click)
    elif event == 2:
        click.append(x)
        click.append(y)
        print(click)
    return click


def resize_grayscale_data(listing, num_keypoint):
    with open('./result.csv', 'a') as f:
        header = []
        header.append('image_name')
        for i in range(num_keypoint):
            header.append('x_point_' + str(i + 1))
            header.append('y_point_' + str(i + 1))
        header.append('count')

        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        i = 0
        for file in listing:
            img = cv2.imread(path1 + '/' + file)
            tk = tkinter.Tk()
            width = tk.winfo_width()
            height = tk.winfo_height()
            scale_width = width / img.shape[1]
            scale_height = height / img.shape[0]
            scale = min(scale_width, scale_height)
            window_width = int(img.shape[1] * scale)
            window_height = int(img.shape[0] * scale)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 250, 1000)

            # set mouse callback function for window
            cv2.setMouseCallback('image', mouse_callback)

            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            count = np.shape(click)[0]
            if count == 0:
                break
            if np.shape(click)[0] < num_keypoint * 2:
                for j in range(num_keypoint * 2 - np.shape(click)[0]):
                    click.append(np.nan)
            data = {'image_name': file}
            for i in range(num_keypoint):
                data.update({'x_point_' + str(i + 1): click[2 * i]})
                data.update({'y_point_' + str(i + 1): click[2 * i + 1]})
            data.update({'count': count / 2})
            data = [data]
            print(data)
            writer.writerows(data)
            click.clear()
            i = i + 1


# =============================================================================
#             print(i)
#             print(count/2)
# =============================================================================
# %%Main

if __name__ == "__main__":
    num_keypoint = 9  # number of keypoint on one image
    path1 = './Keypoint_dataset'  # link of the image folder
    listing = os.listdir(path1)
    num_samples = np.shape(listing)
    resize_grayscale_data(listing, num_keypoint)
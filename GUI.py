import pandas as pd
import PySimpleGUI as sg
import numpy as np
import pickle
import warnings
from sklearn.exceptions import DataConversionWarning
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# 读取模型
with open('ANT_random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# 设置主题为白色背景和红色文本框
sg.theme('LightGrey4')
sg.SetOptions(background_color='white', text_element_background_color='white')

# 所有窗口元素的布局
layout = [
    [sg.Text('Developed by Hang Du, Yuxiao Luo, Kaoshan Dai,Hanrui', font=('Helvetica', 10))],
    [sg.Text('SiChuan University (SCU), Chengdu, China')],
    [sg.Text('Email: duhang202206@163.com')],
    [sg.Frame(layout=[
        [sg.Column([
            [sg.Text('D_tube', size=(10, 1), justification='center'), sg.InputText(key='-f1-', size=(20, 1))],
            [sg.Text('t_tube', size=(10, 1), justification='center'), sg.InputText(key='-f2-', size=(20, 1))],
            [sg.Text('d_bolt', size=(10, 1), justification='center'), sg.InputText(key='-f3-', size=(20, 1))],
            [sg.Text('F_c', size=(10, 1), justification='center'), sg.InputText(key='-f4-', size=(20, 1))],
            [sg.Text('F_a', size=(10, 1), justification='center'), sg.InputText(key='-f5-', size=(20, 1))],
            [sg.Text('F_b', size=(10, 1), justification='center'), sg.InputText(key='-f6-', size=(20, 1))],
            [sg.Text('R_1', size=(10, 1), justification='center'), sg.InputText(key='-f7-', size=(20, 1))],
            [sg.Text('load', size=(10, 1), justification='center'), sg.InputText(key='-f8-', size=(20, 1))],
            [sg.Text('Pretension', size=(10, 1), justification='center'), sg.InputText(key='-f9-', size=(20, 1))],
            [sg.Text('n_bolt', size=(10, 1), justification='center'), sg.InputText(key='-f10-', size=(20, 1))]
        ], size=(220, 260), vertical_scroll_only=True, background_color='white'),
         sg.Image(filename='image.png', key='-Image-', size=(200, 280))],
    ], title='Input parameters', relief=sg.RELIEF_SUNKEN, border_width=1)],
    [sg.Frame(layout=[
        [sg.Image(filename='image1.png', key='-Canvas-', size=(400,240))],  # 调整显示image1.png的大小
        [sg.Text('Status', size=(10, 1)), sg.InputText(key='-OP1-', size=(45, 1))],
    ], title='Output')],
    [sg.Button('Predict'), sg.Button('Cancel')],
    [sg.Canvas(key='-CANVAS-', background_color='white')],
]
# Create the Window
window = sg.Window('Prediction of Stress distribution', layout, finalize=True)

# 禁用 DataConversionWarning 和 UserWarning 警告
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# 在Canvas中显示Matplotlib图表
canvas_elem = window['-CANVAS-']
canvas_elem.Widget.place(x=20, y=800, width=450, height=200)
canvas = FigureCanvasTkAgg(plt.figure(figsize=(8, 3)), master=canvas_elem.Widget)
canvas.get_tk_widget().pack()

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break

    if event == 'Predict':
        if values['-f1-'] == '' or values['-f2-'] == '' or values['-f3-'] == '' or values['-f4-'] == '' or values[
            '-f5-'] == '' or values['-f6-'] == '' or values['-f7-'] == '' or values['-f8-'] == ''or values['-f9-'] == ''\
            or values['-f10-'] == '':
            window['-OP1-'].update('Please fill all the input parameters')
        else:
            x_test = np.array([[float(values['-f1-']), float(values['-f2-']), float(values['-f3-']),
                                float(values['-f4-']), float(values['-f5-']), float(values['-f6-']), float(values['-f7-']),
                                -1*float(values['-f8-']), float(values['-f9-']), float(values['-f10-'])]])
            x_values = [i / 25 * float(values['-f1-']) for i in range(1, 26)]
            y_pred = rf_model.predict(x_test)
            print(y_pred)
            # Separate the first 25 and last 25 values
            y_pred_first_25 = y_pred[0, :25]
            y_pred_last_25 = y_pred[0, 25:]

            # Clear the previous figure and redraw
            plt.clf()
            plt.plot(x_values, y_pred_first_25, marker='o', linestyle='-', label='innerstress')
            plt.plot(x_values, y_pred_last_25, marker='o', linestyle='-', label='outterstress')
            plt.title('Stress Distribution')
            plt.xlabel('Tube height/mm')
            plt.ylabel('Predicted Stress/MPa')
            plt.legend()

            # 保存图表到BytesIO对象
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # 将BytesIO对象的内容保存为图片文件
            image1 = Image.open(buffer)
            image1.save('image1.png')

            # 更新显示image1
            # 更新显示image1
            # 更新显示image1
            window['-Canvas-'].update(filename='image1.png')
            window['-OP1-'].update('finished')
            # 强制刷新整个窗口，以便图像立即更新
            window.Refresh()
window.close()






# 进行预测操作
# ...

# 恢复警告设置
warnings.resetwarnings()
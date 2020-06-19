import PySimpleGUI as sg
import os

sg.theme('Dark Blue 3')
image_elem = sg.Image(filename='')
layout = [[sg.Text('Choose an image to classify: ',font=("Helvetica", 30))],
          [sg.Input(), sg.FileBrowse(key='img')],
          [sg.Button('Classify My Pearl')],
          [image_elem]
]

window = sg.Window('Pearls Classifier', layout, size=(600, 600), auto_size_text=True, auto_size_buttons=True, margins=(30,30), finalize=True)

ImageToClassify = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    try:
        ImageToClassify = str(values["img"])

    except:
        pass
    if event == 'Classify My Pearl':
        if ImageToClassify == '':  #string of the path of the image
            sg.popup('Please choose an image!!!')
        else:
            print(ImageToClassify)
            #image_elem.Update(filename=ImageToClassify)
            print("here is the part to classify the image")
            res = "Your pearl is ***** pearl, congrutulations!!"
            sg.popup(res, title='Result', font=("Helvetica", 25))
window.close()
